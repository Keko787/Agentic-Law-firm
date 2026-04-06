"""
Clause Analyzer
===============
Uses Claude to perform deep legal analysis on each clause by combining:
  - The clause text and classification from the Intake Agent
  - The legal sources and synthesized research from the Research Agent

Produces structured findings with severity ratings, legal basis,
suggested revisions, negotiation strategies, and market comparisons.
"""

from __future__ import annotations
import json
import time
import anthropic
from .models import (
    ClauseAnalysis, Finding, Severity, IssueCategory,
    MissingClause, DocumentScore,
)


# ── Per-clause analysis prompt ────────────────────────────────────────

CLAUSE_ANALYSIS_PROMPT = """You are a senior legal analyst reviewing a contract clause. You have 
access to the clause text, its initial classification, and retrieved legal research.

Produce a thorough legal analysis as a JSON object:

{
  "revised_risk_level": "<'critical' | 'high' | 'medium' | 'low' | 'none'>",
  "findings": [
    {
      "severity": "<'critical' | 'major' | 'moderate' | 'minor' | 'info'>",
      "category": "<'overbroad' | 'one_sided' | 'unenforceable' | 'ambiguous' | 'missing_protection' | 'non_standard' | 'unconscionable' | 'conflict' | 'compliance'>",
      "title": "<short issue headline>",
      "description": "<detailed explanation, 2-3 sentences>",
      "legal_basis": "<specific statute, case law principle, or legal standard supporting this finding>",
      "suggested_revision": "<concrete alternative language or approach>"
    }
  ],
  "overall_assessment": "<2-3 paragraph analysis grounding the findings in the research>",
  "negotiation_strategy": "<practical advice on how to negotiate improvements to this clause>",
  "market_comparison": "<how this clause compares to standard market terms for this type of agreement>"
}

Severity guidelines:
  - CRITICAL: Clause is likely unenforceable, unconscionable, or creates severe legal exposure
  - MAJOR: Significant risk that strongly warrants renegotiation
  - MODERATE: Notable concern worth raising in negotiation
  - MINOR: Suboptimal but low practical impact
  - INFO: Observation with no action needed

Be specific. Cite the legal sources provided. If the research shows the clause deviates from 
legal standards, explain exactly how. Suggested revisions should be concrete enough to propose 
in a redline.

Respond ONLY with the JSON object. No markdown fences."""


# ── Document-level analysis prompt ────────────────────────────────────

DOCUMENT_ANALYSIS_PROMPT = """You are a senior legal analyst performing a document-level review 
of a legal agreement. You have the clause-by-clause analysis results.

Produce a JSON object with:

{
  "missing_clauses": [
    {
      "clause_type": "<type of clause that should be present>",
      "importance": "<'critical' | 'major' | 'moderate' | 'minor'>",
      "description": "<what this clause would protect against>",
      "suggested_language": "<draft language to add, 2-4 sentences>"
    }
  ],
  "document_score": {
    "overall_risk": "<'critical' | 'high' | 'medium' | 'low'>",
    "score": <0-100 integer, 100 = most risky>,
    "summary": "<1-2 sentence overall assessment>",
    "top_concerns": ["<concern 1>", "<concern 2>", ...],
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "proceed_recommendation": "<'sign as-is' | 'sign with minor revisions' | 'negotiate significant changes' | 'do not sign without major revisions' | 'walk away'>"
  },
  "cross_clause_issues": [
    "<issue spanning multiple clauses, e.g. 'The unlimited indemnification in Section 5 contradicts the $100 liability cap in Section 6'>"
  ]
}

Consider standard protective clauses for this document type. For an NDA, expect:
  definitions, confidentiality obligations, exclusions from confidentiality,
  permitted disclosures, return/destruction of materials, term and termination,
  remedies. Flag anything missing.

For document scoring, weigh critical and major findings heavily. A document with 
multiple critical findings should score 70+.

Respond ONLY with the JSON object."""


class ClauseAnalyzer:
    """
    Performs deep legal analysis by merging intake data with research findings.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
    ):
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.max_retries = max_retries

    # ── Per-clause analysis ───────────────────────────────────────────

    def analyze_clause(
        self,
        clause_text: str,
        clause_heading: str,
        clause_type: str,
        risk_level: str,
        intake_flags: list[str],
        intake_summary: str,
        research_context: str,
        research_risk_analysis: str,
        research_recommendations: list[str],
        research_enforceability: str,
        applicable_law: list[str],
    ) -> ClauseAnalysis:
        """
        Analyze a single clause using both intake and research data.
        """
        user_message = self._build_clause_prompt(
            clause_text=clause_text,
            clause_heading=clause_heading,
            clause_type=clause_type,
            risk_level=risk_level,
            intake_flags=intake_flags,
            intake_summary=intake_summary,
            research_context=research_context,
            research_risk_analysis=research_risk_analysis,
            research_recommendations=research_recommendations,
            research_enforceability=research_enforceability,
            applicable_law=applicable_law,
        )

        response = self._call_api(
            system=CLAUSE_ANALYSIS_PROMPT,
            user_message=user_message,
            max_tokens=3000,
        )

        return self._parse_clause_analysis(
            response=response,
            clause_heading=clause_heading,
            clause_type=clause_type,
            original_risk=risk_level,
        )

    # ── Document-level analysis ───────────────────────────────────────

    def analyze_document(
        self,
        document_type: str,
        jurisdiction: str,
        clause_analyses: list[ClauseAnalysis],
        parties: list[str],
    ) -> tuple[list[MissingClause], DocumentScore | None, list[str]]:
        """
        Perform document-level analysis: missing clauses, scoring, cross-clause issues.
        """
        # Summarize clause analyses for the prompt
        clause_summaries = []
        for ca in clause_analyses:
            severity_counts = ca.severity_summary()
            clause_summaries.append(
                f"Clause {ca.clause_index}: {ca.clause_heading} ({ca.clause_type})\n"
                f"  Risk: {ca.revised_risk_level or ca.original_risk_level}\n"
                f"  Findings: {severity_counts}\n"
                f"  Assessment: {ca.overall_assessment[:200]}..."
            )

        user_message = (
            f"DOCUMENT TYPE: {document_type}\n"
            f"JURISDICTION: {jurisdiction or 'Not specified'}\n"
            f"PARTIES: {', '.join(parties) if parties else 'Not specified'}\n\n"
            f"CLAUSE-BY-CLAUSE ANALYSIS:\n\n"
            + "\n\n".join(clause_summaries)
        )

        response = self._call_api(
            system=DOCUMENT_ANALYSIS_PROMPT,
            user_message=user_message,
            max_tokens=3000,
        )

        return self._parse_document_analysis(response)

    # ── Prompt building ───────────────────────────────────────────────

    @staticmethod
    def _build_clause_prompt(
        clause_text: str,
        clause_heading: str,
        clause_type: str,
        risk_level: str,
        intake_flags: list[str],
        intake_summary: str,
        research_context: str,
        research_risk_analysis: str,
        research_recommendations: list[str],
        research_enforceability: str,
        applicable_law: list[str],
    ) -> str:
        flags_str = "\n".join(f"  - {f}" for f in intake_flags) if intake_flags else "  (none)"
        recs_str = "\n".join(f"  - {r}" for r in research_recommendations) if research_recommendations else "  (none)"
        law_str = "\n".join(f"  - {l}" for l in applicable_law) if applicable_law else "  (none)"

        return (
            f"CLAUSE HEADING: {clause_heading}\n"
            f"CLAUSE TYPE: {clause_type}\n"
            f"INITIAL RISK LEVEL: {risk_level}\n\n"
            f"CLAUSE TEXT:\n{clause_text[:2500]}\n\n"
            f"--- INTAKE AGENT FINDINGS ---\n"
            f"Summary: {intake_summary}\n"
            f"Flags:\n{flags_str}\n\n"
            f"--- RESEARCH AGENT FINDINGS ---\n"
            f"Legal context:\n{research_context[:1500]}\n\n"
            f"Risk analysis:\n{research_risk_analysis[:1000]}\n\n"
            f"Enforceability:\n{research_enforceability[:800]}\n\n"
            f"Applicable law:\n{law_str}\n\n"
            f"Research recommendations:\n{recs_str}"
        )

    # ── Response parsing ──────────────────────────────────────────────

    def _parse_clause_analysis(
        self,
        response: str,
        clause_heading: str,
        clause_type: str,
        original_risk: str,
    ) -> ClauseAnalysis:
        """Parse LLM response into a ClauseAnalysis object."""
        data = self._extract_json(response)

        findings = []
        for f in data.get("findings", []):
            try:
                findings.append(Finding(
                    severity=Severity(f.get("severity", "info")),
                    category=IssueCategory(f.get("category", "ambiguous")),
                    title=f.get("title", ""),
                    description=f.get("description", ""),
                    legal_basis=f.get("legal_basis", ""),
                    suggested_revision=f.get("suggested_revision", ""),
                ))
            except (ValueError, KeyError):
                continue

        critical = sum(1 for f in findings if f.severity == Severity.CRITICAL)
        major = sum(1 for f in findings if f.severity == Severity.MAJOR)

        return ClauseAnalysis(
            clause_index=0,  # Set by caller
            clause_heading=clause_heading,
            clause_type=clause_type,
            original_risk_level=original_risk,
            revised_risk_level=data.get("revised_risk_level", original_risk),
            findings=findings,
            overall_assessment=data.get("overall_assessment", ""),
            negotiation_strategy=data.get("negotiation_strategy", ""),
            market_comparison=data.get("market_comparison", ""),
            critical_count=critical,
            major_count=major,
        )

    def _parse_document_analysis(
        self, response: str
    ) -> tuple[list[MissingClause], DocumentScore | None, list[str]]:
        """Parse document-level analysis response."""
        data = self._extract_json(response)

        # Missing clauses
        missing = []
        for mc in data.get("missing_clauses", []):
            try:
                missing.append(MissingClause(
                    clause_type=mc.get("clause_type", ""),
                    importance=Severity(mc.get("importance", "moderate")),
                    description=mc.get("description", ""),
                    suggested_language=mc.get("suggested_language", ""),
                ))
            except (ValueError, KeyError):
                continue

        # Document score
        score_data = data.get("document_score")
        doc_score = None
        if score_data:
            try:
                doc_score = DocumentScore(
                    overall_risk=score_data.get("overall_risk", "medium"),
                    score=int(score_data.get("score", 50)),
                    summary=score_data.get("summary", ""),
                    top_concerns=score_data.get("top_concerns", []),
                    strengths=score_data.get("strengths", []),
                    proceed_recommendation=score_data.get("proceed_recommendation", ""),
                )
            except (ValueError, KeyError):
                pass

        cross_issues = data.get("cross_clause_issues", [])

        return missing, doc_score, cross_issues

    # ── API + JSON helpers ────────────────────────────────────────────

    def _call_api(self, system: str, user_message: str, max_tokens: int = 3000) -> str:
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
                wait = 2 ** attempt * 5
                print(f"  ⏳ Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            except anthropic.APIError as e:
                if attempt < self.max_retries - 1:
                    print(f"  ⚠️  API error (attempt {attempt + 1}): {e}")
                    time.sleep(2)
                else:
                    raise
        raise RuntimeError("Max retries exceeded")

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from LLM response, handling various formats."""
        text = text.strip()
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting from markdown fences
        for fence in ("```json", "```"):
            if fence in text:
                start = text.index(fence) + len(fence)
                end = text.index("```", start) if "```" in text[start:] else len(text)
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass
        # Try finding { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {}
