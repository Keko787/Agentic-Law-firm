"""
Analysis Agent — Main Orchestrator
====================================
The Analysis Agent is Stage 3 of the Agentic Lawyer pipeline.

It takes IntakeResult + ResearchResult and produces an AnalysisResult by:
  1. Matching each clause's research to its intake classification
  2. Running deep per-clause analysis (LLM-powered)
  3. Performing document-level analysis:
     - Missing protective clauses
     - Cross-clause contradictions
     - Overall risk scoring and proceed/walk-away recommendation

This is where the multi-agent value becomes clear: the Analysis Agent
doesn't re-read the document or re-search legal sources. It operates
on the structured outputs of the previous agents, combining them into
a final grounded analysis.

Usage:
    from analysis_agent import AnalysisAgent

    agent = AnalysisAgent(api_key="sk-ant-...")
    result = agent.analyze(intake_result, research_result)
    result.save("output/analysis_result.json")
"""

from __future__ import annotations
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intake_agent.models import IntakeResult, Clause
from research_agent.models import ResearchResult, ClauseResearch
from .models import AnalysisResult, ClauseAnalysis, Severity
from .clause_analyzer import ClauseAnalyzer


console = Console()


class AnalysisAgent:
    """
    Orchestrates deep legal analysis by merging intake + research.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        verbose: bool = True,
    ):
        self.analyzer = ClauseAnalyzer(api_key=api_key, model=model)
        self.verbose = verbose

    def analyze(
        self,
        intake_result: IntakeResult,
        research_result: ResearchResult,
    ) -> AnalysisResult:
        """
        Run the full analysis pipeline.

        Args:
            intake_result: Output from Stage 1 (Intake Agent).
            research_result: Output from Stage 2 (Research Agent).

        Returns:
            AnalysisResult with per-clause findings, missing clauses,
            cross-clause issues, and overall document scoring.
        """
        self._log(f"\n⚖️  Analyzing '{intake_result.metadata.filename}'...")
        self._log(f"   {len(intake_result.clauses)} clauses from intake, "
                   f"{len(research_result.clause_research)} researched")

        # Build research lookup by clause index
        research_by_index = {
            cr.clause_index: cr for cr in research_result.clause_research
        }

        # ── Step 1: Per-clause analysis ───────────────────────────────
        self._log("\n📝 Step 1/2 — Analyzing individual clauses...")
        clause_analyses: list[ClauseAnalysis] = []

        for clause in intake_result.clauses:
            # Skip preamble and signature blocks
            if clause.clause_type.value in ("preamble", "signature_block"):
                self._log(f"   ⏭  Skipping {clause.heading} ({clause.clause_type.value})")
                continue

            research = research_by_index.get(clause.index)

            self._log(f"   🔍 Clause {clause.index}: {clause.heading} "
                       f"({clause.risk_level.value})"
                       f"{' + research' if research else ' (no research)'}")

            analysis = self._analyze_single_clause(clause, research)
            analysis.clause_index = clause.index
            clause_analyses.append(analysis)

            # Quick inline feedback
            sev = analysis.severity_summary()
            if analysis.critical_count:
                self._log(f"      🔴 {analysis.critical_count} critical, "
                           f"{analysis.major_count} major findings",
                           style="red")
            elif analysis.major_count:
                self._log(f"      🟡 {analysis.major_count} major findings",
                           style="yellow")
            else:
                self._log(f"      🟢 {sev}", style="green")

        # ── Step 2: Document-level analysis ───────────────────────────
        self._log("\n📊 Step 2/2 — Document-level analysis...")
        missing, doc_score, cross_issues = self.analyzer.analyze_document(
            document_type=intake_result.metadata.document_type,
            jurisdiction=intake_result.metadata.jurisdiction or "",
            clause_analyses=clause_analyses,
            parties=intake_result.metadata.parties,
        )

        self._log(f"   Missing clauses: {len(missing)}")
        self._log(f"   Cross-clause issues: {len(cross_issues)}")
        if doc_score:
            self._log(f"   Overall risk: {doc_score.overall_risk} "
                       f"(score: {doc_score.score}/100)")
            self._log(f"   Recommendation: {doc_score.proceed_recommendation}")

        # ── Build result ──────────────────────────────────────────────
        result = AnalysisResult(
            document_filename=intake_result.metadata.filename,
            document_type=intake_result.metadata.document_type,
            jurisdiction=intake_result.metadata.jurisdiction or "",
            parties=intake_result.metadata.parties,
            clause_analyses=clause_analyses,
            missing_clauses=missing,
            document_score=doc_score,
            cross_clause_issues=cross_issues,
        )

        if self.verbose:
            self._print_summary(result)

        return result

    # ── Single clause analysis ────────────────────────────────────────

    def _analyze_single_clause(
        self,
        clause: Clause,
        research: ClauseResearch | None,
    ) -> ClauseAnalysis:
        """
        Analyze one clause, merging intake classification with research.
        If research is unavailable, analyze from intake data alone.
        """
        return self.analyzer.analyze_clause(
            clause_text=clause.text,
            clause_heading=clause.heading,
            clause_type=clause.clause_type.value,
            risk_level=clause.risk_level.value,
            intake_flags=clause.flags,
            intake_summary=clause.summary,
            research_context=research.legal_context if research else "",
            research_risk_analysis=research.risk_analysis if research else "",
            research_recommendations=research.recommendations if research else [],
            research_enforceability=research.enforceability_notes if research else "",
            applicable_law=research.applicable_law if research else [],
        )

    # ── Display ───────────────────────────────────────────────────────

    def _log(self, message: str, style: str = "cyan") -> None:
        if self.verbose:
            console.print(message, style=style)

    def _print_summary(self, result: AnalysisResult) -> None:
        """Print formatted analysis summary."""
        console.print()

        # Document score panel
        if result.document_score:
            ds = result.document_score
            risk_color = {
                "critical": "red", "high": "red",
                "medium": "yellow", "low": "green"
            }.get(ds.overall_risk, "white")

            console.print(Panel(
                f"[bold]Risk score: [{risk_color}]{ds.score}/100[/{risk_color}][/bold]\n"
                f"Overall: [{risk_color}]{ds.overall_risk.upper()}[/{risk_color}]\n"
                f"Recommendation: [bold]{ds.proceed_recommendation}[/bold]\n\n"
                f"{ds.summary}",
                title="⚖️  Analysis Agent — Document Assessment",
                border_style=risk_color,
            ))

        # Findings table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", width=4, justify="center")
        table.add_column("Clause", width=22)
        table.add_column("Risk", width=10, justify="center")
        table.add_column("Crit", width=5, justify="center")
        table.add_column("Maj", width=5, justify="center")
        table.add_column("Top finding", width=40)

        risk_colors = {"critical": "red", "high": "red", "medium": "yellow",
                        "low": "green", "none": "dim"}

        for ca in result.clause_analyses:
            risk = ca.revised_risk_level or ca.original_risk_level
            rc = risk_colors.get(risk, "white")
            top = ca.findings[0].title if ca.findings else "—"

            table.add_row(
                str(ca.clause_index),
                ca.clause_heading[:22],
                f"[{rc}]{risk.upper()}[/{rc}]",
                f"[red]{ca.critical_count}[/red]" if ca.critical_count else "0",
                f"[yellow]{ca.major_count}[/yellow]" if ca.major_count else "0",
                top[:40],
            )

        console.print(table)

        # Missing clauses
        if result.missing_clauses:
            console.print(f"\n📋 [bold]Missing protective clauses:[/bold]")
            for mc in result.missing_clauses:
                imp_color = {"critical": "red", "major": "yellow"}.get(
                    mc.importance.value, "white")
                console.print(
                    f"   [{imp_color}]{mc.importance.value.upper()}[/{imp_color}] "
                    f"{mc.clause_type}: {mc.description[:60]}")

        # Cross-clause issues
        if result.cross_clause_issues:
            console.print(f"\n🔗 [bold]Cross-clause issues:[/bold]")
            for issue in result.cross_clause_issues:
                console.print(f"   • {issue}")

        # Severity totals
        totals = result.total_findings_by_severity()
        console.print(f"\n📈 Total findings: ", end="")
        parts = []
        for sev in ["critical", "major", "moderate", "minor", "info"]:
            if totals.get(sev, 0) > 0:
                c = {"critical": "red", "major": "yellow", "moderate": "cyan",
                     "minor": "dim", "info": "dim"}.get(sev, "white")
                parts.append(f"[{c}]{totals[sev]} {sev}[/{c}]")
        console.print(" | ".join(parts) if parts else "none")
