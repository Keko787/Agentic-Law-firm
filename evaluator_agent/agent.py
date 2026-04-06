"""
Evaluator Agent — Main Orchestrator
=====================================
Stage 4 of the Agentic Lawyer pipeline.

The Evaluator is the quality gate. It checks the Analysis Agent's output for:
  1. Grounding: Are findings supported by actual retrieved legal sources?
  2. Consistency: Do findings contradict each other or the research?
  3. Completeness: Were important issues missed?
  4. Hallucination: Did the LLM fabricate legal citations?
  5. Score validity: Does the document risk score match the findings?

If confidence is low on any clause, it triggers a REFLECTION LOOP —
flagging the clause for re-analysis with additional context about
what went wrong.

This is the key differentiator from single-agent systems: the Evaluator
catches errors before the user ever sees them.
"""

from __future__ import annotations
import json
import time
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis_agent.models import AnalysisResult, ClauseAnalysis, Finding
from research_agent.models import ResearchResult, ClauseResearch
from .models import (
    EvaluationResult, ClauseEvaluation, FindingEvaluation,
    ConfidenceLevel,
)

console = Console()

EVAL_PROMPT = """You are a senior legal quality assurance reviewer. Your job is to evaluate 
the quality of a legal analysis produced by an AI system.

You will receive:
- A clause analysis (findings, severity ratings, suggested revisions)
- The research that was available (legal sources, applicable law)
- The original clause text

Evaluate each finding and the overall clause analysis. Produce a JSON object:

{
  "confidence": "<'high' | 'medium' | 'low' | 'failed'>",
  "finding_evaluations": [
    {
      "finding_title": "<title of the finding being evaluated>",
      "is_grounded": <true/false — is this finding supported by the provided research?>,
      "confidence": "<'high' | 'medium' | 'low' | 'failed'>",
      "grounding_notes": "<how well the legal basis matches the research sources>",
      "issues": ["<any quality issue found>"]
    }
  ],
  "consistency_issues": ["<any contradictions between findings, or with the research>"],
  "needs_reanalysis": <true/false>,
  "reanalysis_reason": "<why reanalysis is needed, if applicable>"
}

Evaluation criteria:
  HIGH: Finding is directly supported by retrieved sources, legal basis is accurate,
        severity is proportionate, suggested revision is concrete and actionable
  MEDIUM: Finding is reasonable but legal basis is partially supported or generic,
          or the severity could be debated
  LOW: Finding lacks clear support in the retrieved sources, or the legal basis
       appears fabricated or misapplied
  FAILED: Finding contradicts the research, cites non-existent law, or contains
          obvious errors

Flag needs_reanalysis=true if ANY finding scores 'failed' or if 2+ score 'low'.

Respond ONLY with the JSON object."""


SCORE_VALIDATION_PROMPT = """You are validating a document risk score. Given the clause-level 
findings summary, evaluate whether the overall score is appropriate.

Respond with a JSON object:
{
  "score_validated": <true/false>,
  "score_adjustment": "<'' if valid, or 'should be higher/lower because...' if not>",
  "hallucination_flags": ["<any findings across all clauses that appear fabricated>"],
  "cross_clause_validation": ["<validated or disputed cross-clause issues>"]
}

Respond ONLY with the JSON object."""


class EvaluatorAgent:
    """
    Quality gate that validates Analysis Agent output.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        verbose: bool = True,
    ):
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.max_retries = max_retries
        self.verbose = verbose

    def evaluate(
        self,
        analysis_result: AnalysisResult,
        research_result: ResearchResult,
    ) -> EvaluationResult:
        """
        Evaluate the quality of an AnalysisResult.

        Returns an EvaluationResult with confidence scores, grounding checks,
        and reflection loop triggers.
        """
        self._log("\n🔎 Evaluating analysis quality...")

        research_by_index = {
            cr.clause_index: cr for cr in research_result.clause_research
        }

        # ── Evaluate each clause ──────────────────────────────────────
        self._log("   Step 1/2 — Per-clause evaluation...")
        clause_evals: list[ClauseEvaluation] = []
        reanalysis_needed: list[int] = []

        for ca in analysis_result.clause_analyses:
            research = research_by_index.get(ca.clause_index)

            self._log(f"   Checking clause {ca.clause_index}: {ca.clause_heading}...")
            eval_result = self._evaluate_clause(ca, research)
            clause_evals.append(eval_result)

            if eval_result.needs_reanalysis:
                reanalysis_needed.append(ca.clause_index)
                self._log(f"      🔄 Needs reanalysis: {eval_result.reanalysis_reason}",
                           style="yellow")
            else:
                conf_color = {"high": "green", "medium": "cyan",
                              "low": "yellow", "failed": "red"}.get(
                    eval_result.confidence.value, "white")
                self._log(f"      [{conf_color}]Confidence: "
                           f"{eval_result.confidence.value}[/{conf_color}]",
                           style=conf_color)

        # ── Document-level validation ─────────────────────────────────
        self._log("   Step 2/2 — Document-level validation...")
        score_valid, score_adj, hallucinations, cross_valid = \
            self._validate_document_level(analysis_result)

        # ── Determine overall confidence ──────────────────────────────
        if any(ce.confidence == ConfidenceLevel.FAILED for ce in clause_evals):
            overall = ConfidenceLevel.LOW
        elif len(reanalysis_needed) > 0:
            overall = ConfidenceLevel.MEDIUM
        elif all(ce.confidence == ConfidenceLevel.HIGH for ce in clause_evals):
            overall = ConfidenceLevel.HIGH
        else:
            overall = ConfidenceLevel.MEDIUM

        result = EvaluationResult(
            document_filename=analysis_result.document_filename,
            overall_confidence=overall,
            clause_evaluations=clause_evals,
            score_validated=score_valid,
            score_adjustment=score_adj,
            cross_clause_validation=cross_valid,
            hallucination_flags=hallucinations,
            clauses_needing_reanalysis=reanalysis_needed,
            reflection_notes=[
                f"Clause {idx} flagged for reanalysis" for idx in reanalysis_needed
            ] if reanalysis_needed else ["All clauses passed evaluation"],
        )

        if self.verbose:
            self._print_summary(result)

        return result

    # ── Per-clause evaluation ─────────────────────────────────────────

    def _evaluate_clause(
        self,
        clause_analysis: ClauseAnalysis,
        research: ClauseResearch | None,
    ) -> ClauseEvaluation:
        """Evaluate a single clause's analysis against its research."""

        # Build findings summary for the prompt
        findings_text = ""
        for i, f in enumerate(clause_analysis.findings):
            findings_text += (
                f"\nFinding {i + 1}: {f.title}\n"
                f"  Severity: {f.severity.value}\n"
                f"  Category: {f.category.value}\n"
                f"  Description: {f.description}\n"
                f"  Legal basis: {f.legal_basis}\n"
                f"  Suggested revision: {f.suggested_revision}\n"
            )

        # Build research context
        research_text = "(No research available for this clause)"
        if research:
            sources_summary = "\n".join(
                f"  - {s.title}: {s.text[:200]}..."
                for s in research.sources[:5]
            )
            research_text = (
                f"Legal context: {research.legal_context[:500]}\n"
                f"Applicable law: {', '.join(research.applicable_law)}\n"
                f"Enforceability: {research.enforceability_notes[:300]}\n"
                f"Sources:\n{sources_summary}"
            )

        user_message = (
            f"CLAUSE: {clause_analysis.clause_heading} ({clause_analysis.clause_type})\n"
            f"ORIGINAL TEXT: (clause index {clause_analysis.clause_index})\n\n"
            f"ANALYSIS FINDINGS:\n{findings_text}\n\n"
            f"OVERALL ASSESSMENT:\n{clause_analysis.overall_assessment[:500]}\n\n"
            f"AVAILABLE RESEARCH:\n{research_text}"
        )

        response = self._call_api(EVAL_PROMPT, user_message, max_tokens=2000)
        data = self._extract_json(response)

        # Parse finding evaluations
        finding_evals = []
        for fe in data.get("finding_evaluations", []):
            try:
                finding_evals.append(FindingEvaluation(
                    finding_title=fe.get("finding_title", ""),
                    is_grounded=fe.get("is_grounded", True),
                    confidence=ConfidenceLevel(fe.get("confidence", "medium")),
                    grounding_notes=fe.get("grounding_notes", ""),
                    issues=fe.get("issues", []),
                ))
            except (ValueError, KeyError):
                continue

        try:
            confidence = ConfidenceLevel(data.get("confidence", "medium"))
        except ValueError:
            confidence = ConfidenceLevel.MEDIUM

        return ClauseEvaluation(
            clause_index=clause_analysis.clause_index,
            clause_heading=clause_analysis.clause_heading,
            confidence=confidence,
            finding_evaluations=finding_evals,
            consistency_issues=data.get("consistency_issues", []),
            needs_reanalysis=data.get("needs_reanalysis", False),
            reanalysis_reason=data.get("reanalysis_reason", ""),
        )

    # ── Document-level validation ─────────────────────────────────────

    def _validate_document_level(
        self, analysis: AnalysisResult
    ) -> tuple[bool, str, list[str], list[str]]:
        """Validate the document score and cross-clause issues."""

        # Build summary for the prompt
        clause_summary = "\n".join(
            f"Clause {ca.clause_index} ({ca.clause_heading}): "
            f"risk={ca.revised_risk_level or ca.original_risk_level}, "
            f"critical={ca.critical_count}, major={ca.major_count}, "
            f"findings={len(ca.findings)}"
            for ca in analysis.clause_analyses
        )

        score_info = ""
        if analysis.document_score:
            ds = analysis.document_score
            score_info = (
                f"Score: {ds.score}/100, Risk: {ds.overall_risk}\n"
                f"Recommendation: {ds.proceed_recommendation}\n"
                f"Top concerns: {ds.top_concerns}\n"
            )

        cross_info = "\n".join(
            f"  - {issue}" for issue in analysis.cross_clause_issues
        ) if analysis.cross_clause_issues else "(none)"

        user_message = (
            f"CLAUSE SUMMARIES:\n{clause_summary}\n\n"
            f"DOCUMENT SCORE:\n{score_info}\n\n"
            f"CROSS-CLAUSE ISSUES:\n{cross_info}\n\n"
            f"MISSING CLAUSES: {len(analysis.missing_clauses)}"
        )

        response = self._call_api(SCORE_VALIDATION_PROMPT, user_message, max_tokens=1500)
        data = self._extract_json(response)

        return (
            data.get("score_validated", True),
            data.get("score_adjustment", ""),
            data.get("hallucination_flags", []),
            data.get("cross_clause_validation", []),
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _call_api(self, system: str, user_message: str, max_tokens: int = 2000) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user_message}],
                )
                return response.content[0].text
            except anthropic.RateLimitError:
                time.sleep(2 ** attempt * 5)
            except anthropic.APIError as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                else:
                    raise
        raise RuntimeError("Max retries exceeded")

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {}

    def _log(self, msg: str, style: str = "cyan") -> None:
        if self.verbose:
            console.print(msg, style=style)

    def _print_summary(self, result: EvaluationResult) -> None:
        conf_color = {"high": "green", "medium": "yellow",
                      "low": "red", "failed": "red"}.get(
            result.overall_confidence.value, "white")

        console.print(Panel(
            f"[bold]Overall confidence: "
            f"[{conf_color}]{result.overall_confidence.value.upper()}[/{conf_color}][/bold]\n"
            f"Pass rate: {result.pass_rate:.0%}\n"
            f"Score validated: {'Yes' if result.score_validated else 'No — ' + result.score_adjustment}\n"
            f"Hallucination flags: {len(result.hallucination_flags)}\n"
            f"Clauses needing reanalysis: {len(result.clauses_needing_reanalysis)}",
            title="🔎 Evaluator Agent — Quality Report",
            border_style=conf_color,
        ))

        table = Table(show_header=True, header_style="bold green")
        table.add_column("#", width=4, justify="center")
        table.add_column("Clause", width=22)
        table.add_column("Confidence", width=12, justify="center")
        table.add_column("Grounded", width=10, justify="center")
        table.add_column("Issues", width=35)

        for ce in result.clause_evaluations:
            cc = {"high": "green", "medium": "yellow",
                  "low": "red", "failed": "red"}.get(ce.confidence.value, "white")
            grounded_count = sum(1 for fe in ce.finding_evaluations if fe.is_grounded)
            total = len(ce.finding_evaluations)
            issues = "; ".join(ce.consistency_issues[:2]) if ce.consistency_issues else "—"

            table.add_row(
                str(ce.clause_index),
                ce.clause_heading[:22],
                f"[{cc}]{ce.confidence.value.upper()}[/{cc}]",
                f"{grounded_count}/{total}" if total else "—",
                issues[:35],
            )

        console.print(table)

        if result.hallucination_flags:
            console.print(f"\n⚠️  [bold red]Hallucination flags:[/bold red]")
            for h in result.hallucination_flags:
                console.print(f"   • {h}")
