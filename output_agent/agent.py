"""
Output Agent — Main Orchestrator
==================================
Stage 5 (final) of the Agentic Lawyer pipeline.

Takes the validated AnalysisResult + EvaluationResult and produces
a structured, plain-language report suitable for non-lawyers.

The report includes:
  - Executive summary with proceed/walk-away recommendation
  - Risk scorecard
  - Clause-by-clause findings (prioritized by severity)
  - Missing clause recommendations
  - Negotiation playbook
  - Confidence disclosure

Two output formats:
  - Structured JSON (FinalReport) for programmatic consumption
  - Markdown report for human reading / PDF conversion
"""

from __future__ import annotations
import json
import time
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis_agent.models import AnalysisResult, ClauseAnalysis, Severity
from evaluator_agent.models import EvaluationResult, ConfidenceLevel
from .models import FinalReport, ReportSection

console = Console()

EXECUTIVE_SUMMARY_PROMPT = """You are a legal communication specialist. Write an executive summary 
of a legal document review for a non-lawyer audience.

You will receive the analysis results. Write a clear, actionable executive summary that:
1. States what type of document was reviewed and between which parties
2. Gives the bottom-line recommendation (sign / negotiate / walk away)
3. Highlights the top 3-5 concerns in plain language
4. Notes any strengths of the document
5. States the overall risk level and confidence

Write 3-4 paragraphs. Use plain language — avoid legal jargon where possible, and explain 
any legal terms you use. This should be readable by a small business owner or freelancer.

Respond with ONLY the executive summary text. No JSON, no markdown headers."""


class OutputAgent:
    """
    Generates the final report from pipeline outputs.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        verbose: bool = True,
    ):
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.verbose = verbose

    def generate_report(
        self,
        analysis_result: AnalysisResult,
        evaluation_result: EvaluationResult,
    ) -> FinalReport:
        """
        Generate the final report combining analysis and evaluation.
        """
        self._log("\n📄 Generating final report...")

        # ── Executive summary (LLM-generated) ─────────────────────────
        self._log("   Writing executive summary...")
        executive_summary = self._generate_executive_summary(
            analysis_result, evaluation_result
        )

        # ── Build report sections ─────────────────────────────────────
        self._log("   Building report sections...")
        sections = []

        # Risk scorecard
        sections.append(self._build_scorecard(analysis_result))

        # Clause-by-clause findings (sorted by severity)
        sorted_clauses = sorted(
            analysis_result.clause_analyses,
            key=lambda ca: {"critical": 0, "high": 1, "medium": 2,
                           "low": 3, "none": 4}.get(
                ca.revised_risk_level or ca.original_risk_level, 4)
        )
        for ca in sorted_clauses:
            sections.append(self._build_clause_section(ca, evaluation_result))

        # Missing clauses
        if analysis_result.missing_clauses:
            sections.append(self._build_missing_section(analysis_result))

        # Cross-clause issues
        if analysis_result.cross_clause_issues:
            sections.append(self._build_cross_clause_section(analysis_result))

        # Negotiation playbook
        sections.append(self._build_negotiation_playbook(analysis_result))

        # Confidence disclosure
        sections.append(self._build_confidence_disclosure(evaluation_result))

        # ── Build markdown report ─────────────────────────────────────
        self._log("   Compiling markdown report...")
        markdown = self._compile_markdown(
            analysis_result, evaluation_result,
            executive_summary, sections
        )

        # ── Assemble final report ─────────────────────────────────────
        totals = analysis_result.total_findings_by_severity()

        report = FinalReport(
            document_filename=analysis_result.document_filename,
            document_type=analysis_result.document_type,
            jurisdiction=analysis_result.jurisdiction,
            parties=analysis_result.parties,
            executive_summary=executive_summary,
            risk_score=analysis_result.document_score.score if analysis_result.document_score else 0,
            risk_level=analysis_result.document_score.overall_risk if analysis_result.document_score else "",
            proceed_recommendation=analysis_result.document_score.proceed_recommendation if analysis_result.document_score else "",
            sections=sections,
            report_markdown=markdown,
            pipeline_confidence=evaluation_result.overall_confidence.value,
            clauses_analyzed=len(analysis_result.clause_analyses),
            total_findings=sum(totals.values()),
            critical_findings=totals.get("critical", 0),
        )

        if self.verbose:
            self._print_summary(report)

        return report

    # ── Section builders ──────────────────────────────────────────────

    def _generate_executive_summary(
        self,
        analysis: AnalysisResult,
        evaluation: EvaluationResult,
    ) -> str:
        """Generate plain-language executive summary via LLM."""
        ds = analysis.document_score
        totals = analysis.total_findings_by_severity()

        summary_data = (
            f"Document: {analysis.document_filename} ({analysis.document_type})\n"
            f"Parties: {', '.join(analysis.parties)}\n"
            f"Jurisdiction: {analysis.jurisdiction or 'Not specified'}\n\n"
        )

        if ds:
            summary_data += (
                f"Risk score: {ds.score}/100 ({ds.overall_risk})\n"
                f"Recommendation: {ds.proceed_recommendation}\n"
                f"Top concerns: {'; '.join(ds.top_concerns)}\n"
                f"Strengths: {'; '.join(ds.strengths)}\n\n"
            )

        summary_data += (
            f"Findings: {totals.get('critical', 0)} critical, "
            f"{totals.get('major', 0)} major, "
            f"{totals.get('moderate', 0)} moderate\n"
            f"Missing clauses: {len(analysis.missing_clauses)}\n"
            f"Cross-clause issues: {len(analysis.cross_clause_issues)}\n"
            f"Evaluation confidence: {evaluation.overall_confidence.value}\n\n"
        )

        # Add top findings
        critical_findings = analysis.critical_findings()
        if critical_findings:
            summary_data += "Critical findings:\n"
            for heading, finding in critical_findings[:5]:
                summary_data += f"  - {heading}: {finding.title} — {finding.description[:150]}\n"

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                system=EXECUTIVE_SUMMARY_PROMPT,
                messages=[{"role": "user", "content": summary_data}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Executive summary generation failed: {e}"

    def _build_scorecard(self, analysis: AnalysisResult) -> ReportSection:
        ds = analysis.document_score
        totals = analysis.total_findings_by_severity()

        content = ""
        if ds:
            content += f"**Overall risk:** {ds.overall_risk.upper()} ({ds.score}/100)\n\n"
            content += f"**Recommendation:** {ds.proceed_recommendation}\n\n"
        content += (
            f"**Findings breakdown:** "
            f"{totals.get('critical', 0)} critical, "
            f"{totals.get('major', 0)} major, "
            f"{totals.get('moderate', 0)} moderate, "
            f"{totals.get('minor', 0)} minor\n\n"
            f"**Clauses analyzed:** {len(analysis.clause_analyses)}\n\n"
            f"**Missing clauses:** {len(analysis.missing_clauses)}"
        )

        return ReportSection(heading="Risk scorecard", content=content, priority=0)

    def _build_clause_section(
        self, ca: ClauseAnalysis, evaluation: EvaluationResult
    ) -> ReportSection:
        risk = ca.revised_risk_level or ca.original_risk_level
        risk_badge = {"critical": "🔴", "high": "🔴", "medium": "🟡",
                      "low": "🟢", "none": "⚪"}.get(risk, "⚪")

        content = f"{risk_badge} **Risk:** {risk.upper()}\n\n"

        if ca.overall_assessment:
            content += f"{ca.overall_assessment}\n\n"

        if ca.findings:
            content += "**Findings:**\n\n"
            for f in ca.findings:
                sev_badge = {"critical": "🔴", "major": "🟠", "moderate": "🟡",
                             "minor": "⚪", "info": "ℹ️"}.get(f.severity.value, "")
                content += (
                    f"- {sev_badge} **{f.title}** ({f.severity.value})\n"
                    f"  {f.description}\n"
                )
                if f.legal_basis:
                    content += f"  *Legal basis:* {f.legal_basis}\n"
                if f.suggested_revision:
                    content += f"  *Suggested revision:* {f.suggested_revision}\n"
                content += "\n"

        if ca.market_comparison:
            content += f"**Market comparison:** {ca.market_comparison}\n\n"

        if ca.negotiation_strategy:
            content += f"**Negotiation strategy:** {ca.negotiation_strategy}\n"

        return ReportSection(
            heading=f"{ca.clause_heading} (Section {ca.clause_index})",
            content=content,
            priority={"critical": 1, "high": 2, "medium": 3,
                      "low": 4, "none": 5}.get(risk, 5),
        )

    def _build_missing_section(self, analysis: AnalysisResult) -> ReportSection:
        content = ""
        for mc in analysis.missing_clauses:
            imp_badge = {"critical": "🔴", "major": "🟠", "moderate": "🟡",
                         "minor": "⚪"}.get(mc.importance.value, "")
            content += (
                f"- {imp_badge} **{mc.clause_type}** ({mc.importance.value})\n"
                f"  {mc.description}\n"
            )
            if mc.suggested_language:
                content += f"  *Suggested language:* {mc.suggested_language}\n"
            content += "\n"

        return ReportSection(
            heading="Missing protective clauses",
            content=content,
            priority=6,
        )

    def _build_cross_clause_section(self, analysis: AnalysisResult) -> ReportSection:
        content = ""
        for issue in analysis.cross_clause_issues:
            content += f"- {issue}\n\n"
        return ReportSection(
            heading="Cross-clause issues",
            content=content,
            priority=7,
        )

    def _build_negotiation_playbook(self, analysis: AnalysisResult) -> ReportSection:
        content = "Prioritized list of items to negotiate, ordered by severity:\n\n"

        priority_items = []
        for ca in analysis.clause_analyses:
            for f in ca.findings:
                if f.severity in (Severity.CRITICAL, Severity.MAJOR) and f.suggested_revision:
                    priority_items.append((f.severity.value, ca.clause_heading, f.title, f.suggested_revision))

        priority_items.sort(key=lambda x: {"critical": 0, "major": 1}.get(x[0], 2))

        for i, (sev, heading, title, revision) in enumerate(priority_items, 1):
            badge = "🔴" if sev == "critical" else "🟠"
            content += f"{i}. {badge} **{heading} — {title}**\n   {revision}\n\n"

        if not priority_items:
            content += "No critical or major items requiring negotiation.\n"

        return ReportSection(heading="Negotiation playbook", content=content, priority=8)

    def _build_confidence_disclosure(self, evaluation: EvaluationResult) -> ReportSection:
        content = (
            f"**Pipeline confidence:** {evaluation.overall_confidence.value.upper()}\n\n"
            f"**Pass rate:** {evaluation.pass_rate:.0%} of clauses passed quality checks\n\n"
            f"**Score validated:** {'Yes' if evaluation.score_validated else 'No'}\n"
        )

        if evaluation.score_adjustment:
            content += f"\n**Score adjustment note:** {evaluation.score_adjustment}\n"

        if evaluation.hallucination_flags:
            content += "\n**Grounding warnings:** The following findings may not be fully supported by retrieved legal sources:\n\n"
            for h in evaluation.hallucination_flags:
                content += f"- {h}\n"

        content += (
            "\n\n*This analysis was generated by an AI system. It is not legal advice. "
            "Consult a licensed attorney before making legal decisions based on this report.*"
        )

        return ReportSection(heading="Confidence and disclaimers", content=content, priority=99)

    # ── Markdown compilation ──────────────────────────────────────────

    def _compile_markdown(
        self,
        analysis: AnalysisResult,
        evaluation: EvaluationResult,
        executive_summary: str,
        sections: list[ReportSection],
    ) -> str:
        """Compile all sections into a single markdown report."""
        lines = [
            f"# Legal Document Review: {analysis.document_filename}",
            "",
            f"**Document type:** {analysis.document_type}  ",
            f"**Parties:** {', '.join(analysis.parties) if analysis.parties else 'Not specified'}  ",
            f"**Jurisdiction:** {analysis.jurisdiction or 'Not specified'}  ",
            "",
            "---",
            "",
            "## Executive summary",
            "",
            executive_summary,
            "",
            "---",
            "",
        ]

        for section in sorted(sections, key=lambda s: s.priority):
            lines.append(f"## {section.heading}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    # ── Display ───────────────────────────────────────────────────────

    def _log(self, msg: str, style: str = "cyan") -> None:
        if self.verbose:
            console.print(msg, style=style)

    def _print_summary(self, report: FinalReport) -> None:
        risk_color = {"critical": "red", "high": "red",
                      "medium": "yellow", "low": "green"}.get(report.risk_level, "white")

        console.print(Panel(
            f"[bold]{report.document_filename}[/bold]\n\n"
            f"Risk: [{risk_color}]{report.risk_level.upper()} "
            f"({report.risk_score}/100)[/{risk_color}]\n"
            f"Recommendation: [bold]{report.proceed_recommendation}[/bold]\n"
            f"Confidence: {report.pipeline_confidence}\n\n"
            f"Findings: {report.total_findings} total, "
            f"{report.critical_findings} critical\n"
            f"Report sections: {len(report.sections)}",
            title="📄 Output Agent — Final Report",
            border_style="green",
        ))
