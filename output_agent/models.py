"""
Data models for the Output Agent.

The Output Agent produces the final deliverable — a structured,
plain-language report combining all pipeline outputs.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


class ReportSection(BaseModel):
    """A single section of the final report."""
    heading: str
    content: str
    priority: int = Field(default=0, description="Sort order, lower = higher priority")


class FinalReport(BaseModel):
    """The final deliverable from the pipeline."""
    document_filename: str
    document_type: str
    jurisdiction: str = ""
    parties: list[str] = Field(default_factory=list)

    # Report content
    executive_summary: str = Field(default="", description="2-3 paragraph overview for decision-makers")
    risk_score: int = Field(default=0)
    risk_level: str = Field(default="")
    proceed_recommendation: str = Field(default="")

    sections: list[ReportSection] = Field(default_factory=list)
    report_markdown: str = Field(default="", description="Full report as markdown")

    # Metadata
    pipeline_confidence: str = Field(default="", description="Evaluator confidence level")
    clauses_analyzed: int = Field(default=0)
    total_findings: int = Field(default=0)
    critical_findings: int = Field(default=0)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    def save_markdown(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.report_markdown)
