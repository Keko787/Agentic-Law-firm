"""
Data models for the Analysis Agent pipeline.

The Analysis Agent merges intake classification with research findings
to produce grounded, actionable legal analysis:

  ClauseAnalysis   → deep analysis of one clause
  MissingClause    → a protective clause the document lacks
  DocumentScore    → overall risk scoring
  AnalysisResult   → full output bundle
"""

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class Severity(str, Enum):
    """Issue severity for individual findings."""
    CRITICAL  = "critical"    # Likely unenforceable or severely prejudicial
    MAJOR     = "major"       # Significant risk, strongly recommend change
    MODERATE  = "moderate"    # Notable concern, negotiation recommended
    MINOR     = "minor"       # Suboptimal but low practical impact
    INFO      = "info"        # Informational observation


class IssueCategory(str, Enum):
    """What kind of problem a finding represents."""
    OVERBROAD         = "overbroad"           # Scope too wide
    ONE_SIDED          = "one_sided"           # Favors one party disproportionately
    UNENFORCEABLE      = "unenforceable"       # Likely invalid under applicable law
    AMBIGUOUS          = "ambiguous"           # Unclear language creates uncertainty
    MISSING_PROTECTION = "missing_protection"  # Expected safeguard absent
    NON_STANDARD       = "non_standard"        # Deviates from market practice
    UNCONSCIONABLE     = "unconscionable"      # May be struck by a court
    CONFLICT           = "conflict"            # Internal contradiction with other clauses
    COMPLIANCE         = "compliance"          # Potential regulatory issue


class Finding(BaseModel):
    """A single issue or observation about a clause."""
    severity: Severity
    category: IssueCategory
    title: str                              = Field(description="Short issue headline")
    description: str                        = Field(description="Detailed explanation of the issue")
    legal_basis: str                        = Field(default="", description="Legal authority supporting this finding")
    suggested_revision: str                 = Field(default="", description="Concrete alternative language or approach")


class ClauseAnalysis(BaseModel):
    """
    Deep analysis of a single clause, combining intake + research.
    """
    clause_index: int
    clause_heading: str
    clause_type: str
    original_risk_level: str                = Field(description="Risk from intake classification")
    revised_risk_level: str                 = Field(default="", description="Risk after research-grounded analysis")

    # Core analysis
    findings: list[Finding]                 = Field(default_factory=list)
    overall_assessment: str                 = Field(default="", description="2-3 paragraph grounded analysis")
    negotiation_strategy: str               = Field(default="", description="How to approach negotiating this clause")
    market_comparison: str                  = Field(default="", description="How this compares to standard market terms")

    # Quick reference
    critical_count: int                     = Field(default=0)
    major_count: int                        = Field(default=0)

    def severity_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.severity.value] = counts.get(f.severity.value, 0) + 1
        return counts


class MissingClause(BaseModel):
    """A protective clause that should be present but isn't."""
    clause_type: str                        = Field(description="What type of clause is missing")
    importance: Severity                    = Field(description="How important is this omission")
    description: str                        = Field(description="What this clause would protect against")
    suggested_language: str                 = Field(default="", description="Draft language to add")


class DocumentScore(BaseModel):
    """Overall risk scoring for the document."""
    overall_risk: str                       = Field(description="critical / high / medium / low")
    score: int                              = Field(description="0-100 risk score, 100 = most risky")
    summary: str                            = Field(description="1-2 sentence overall assessment")
    top_concerns: list[str]                 = Field(default_factory=list, description="Top 3-5 concerns")
    strengths: list[str]                    = Field(default_factory=list, description="Positive aspects of the document")
    proceed_recommendation: str             = Field(default="", description="Sign / negotiate / walk away")


class AnalysisResult(BaseModel):
    """
    Complete output of the Analysis Agent.
    Passed downstream to the Evaluator Agent and Output Agent.
    """
    document_filename: str
    document_type: str
    jurisdiction: str                       = Field(default="")
    parties: list[str]                      = Field(default_factory=list)

    # Per-clause analysis
    clause_analyses: list[ClauseAnalysis]   = Field(default_factory=list)

    # Document-level findings
    missing_clauses: list[MissingClause]    = Field(default_factory=list)
    document_score: Optional[DocumentScore] = None
    cross_clause_issues: list[str]          = Field(default_factory=list, description="Issues spanning multiple clauses")

    processing_notes: list[str]             = Field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    def critical_findings(self) -> list[tuple[str, Finding]]:
        """Return all critical findings with their clause heading."""
        results = []
        for ca in self.clause_analyses:
            for f in ca.findings:
                if f.severity == Severity.CRITICAL:
                    results.append((ca.clause_heading, f))
        return results

    def total_findings_by_severity(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ca in self.clause_analyses:
            for f in ca.findings:
                counts[f.severity.value] = counts.get(f.severity.value, 0) + 1
        return counts
