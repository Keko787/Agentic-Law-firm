"""
Data models for the Evaluator Agent.

The Evaluator checks quality, grounding, and consistency of the
Analysis Agent's output. It produces confidence scores and can
trigger re-analysis via the reflection loop.
"""

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    HIGH   = "high"       # Well-grounded, consistent, actionable
    MEDIUM = "medium"     # Mostly grounded, minor gaps
    LOW    = "low"        # Weak grounding or inconsistencies found
    FAILED = "failed"     # Major quality issues — needs re-analysis


class FindingEvaluation(BaseModel):
    """Evaluation of a single finding from the analysis."""
    finding_title: str
    is_grounded: bool                       = Field(description="Is the finding supported by retrieved sources?")
    confidence: ConfidenceLevel
    grounding_notes: str                    = Field(default="", description="How well the legal basis checks out")
    issues: list[str]                       = Field(default_factory=list, description="Quality issues found")


class ClauseEvaluation(BaseModel):
    """Evaluation of a single clause's analysis."""
    clause_index: int
    clause_heading: str
    confidence: ConfidenceLevel
    finding_evaluations: list[FindingEvaluation] = Field(default_factory=list)
    consistency_issues: list[str]           = Field(default_factory=list)
    needs_reanalysis: bool                  = Field(default=False)
    reanalysis_reason: str                  = Field(default="")


class EvaluationResult(BaseModel):
    """Complete output of the Evaluator Agent."""
    document_filename: str
    overall_confidence: ConfidenceLevel
    clause_evaluations: list[ClauseEvaluation] = Field(default_factory=list)

    # Document-level checks
    score_validated: bool                   = Field(default=True)
    score_adjustment: str                   = Field(default="", description="Suggested score change if any")
    cross_clause_validation: list[str]      = Field(default_factory=list, description="Validated or disputed cross-clause issues")
    hallucination_flags: list[str]          = Field(default_factory=list, description="Findings that appear ungrounded")

    # Reflection loop
    clauses_needing_reanalysis: list[int]   = Field(default_factory=list)
    reflection_notes: list[str]             = Field(default_factory=list)

    processing_notes: list[str]             = Field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @property
    def pass_rate(self) -> float:
        if not self.clause_evaluations:
            return 0.0
        passed = sum(1 for ce in self.clause_evaluations
                     if ce.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM))
        return passed / len(self.clause_evaluations)
