"""
Data models for the Intake Agent pipeline.

These models define the structured output at each stage:
  RawPage      → raw text per PDF page
  Clause       → a segmented clause with classification
  IntakeResult → the full output bundle from the Intake Agent
"""

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
import json


# ── Clause types the classifier can assign ────────────────────────────
class ClauseType(str, Enum):
    """
    Legal clause categories relevant to contracts and NDAs.
    Expand this enum as you add more document types.
    """
    DEFINITIONS        = "definitions"
    CONFIDENTIALITY    = "confidentiality"
    NON_DISCLOSURE     = "non_disclosure"
    OBLIGATIONS        = "obligations"
    INDEMNIFICATION    = "indemnification"
    TERMINATION        = "termination"
    GOVERNING_LAW      = "governing_law"
    DISPUTE_RESOLUTION = "dispute_resolution"
    LIABILITY          = "liability"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    REPRESENTATIONS    = "representations"
    NON_COMPETE        = "non_compete"
    NON_SOLICITATION   = "non_solicitation"
    PAYMENT_TERMS      = "payment_terms"
    SCOPE_OF_WORK      = "scope_of_work"
    FORCE_MAJEURE      = "force_majeure"
    ASSIGNMENT         = "assignment"
    SEVERABILITY       = "severability"
    ENTIRE_AGREEMENT   = "entire_agreement"
    AMENDMENTS         = "amendments"
    NOTICES            = "notices"
    PREAMBLE           = "preamble"
    SIGNATURE_BLOCK    = "signature_block"
    MISCELLANEOUS      = "miscellaneous"
    UNKNOWN            = "unknown"


class RiskLevel(str, Enum):
    """How risky a clause is from the reviewing party's perspective."""
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    NONE   = "none"    # not applicable (e.g., preamble)


# ── Per-page raw extraction ───────────────────────────────────────────
class RawPage(BaseModel):
    """Raw text extracted from a single PDF page."""
    page_number: int
    text: str


# ── A single segmented + classified clause ────────────────────────────
class Clause(BaseModel):
    """
    One logical clause extracted from the document.
    Populated in two passes:
      1. Segmentation (text, heading, index, source_pages)
      2. Classification (clause_type, risk_level, summary, key_terms)
    """
    index: int                              = Field(description="0-based position in the document")
    heading: str                            = Field(description="Section heading or generated label")
    text: str                               = Field(description="Full clause text")
    clause_type: ClauseType                 = Field(default=ClauseType.UNKNOWN)
    risk_level: RiskLevel                   = Field(default=RiskLevel.NONE)
    summary: str                            = Field(default="", description="Plain-language one-liner")
    key_terms: list[str]                    = Field(default_factory=list, description="Important legal terms found")
    flags: list[str]                        = Field(default_factory=list, description="Potential issues or concerns")
    source_pages: list[int]                 = Field(default_factory=list, description="PDF pages this clause spans")


# ── Document metadata ─────────────────────────────────────────────────
class DocumentMetadata(BaseModel):
    """High-level document identification."""
    filename: str
    total_pages: int
    document_type: str                      = Field(default="unknown", description="e.g. NDA, freelance_contract, lease")
    parties: list[str]                      = Field(default_factory=list, description="Named parties in the agreement")
    effective_date: Optional[str]           = None
    jurisdiction: Optional[str]             = None


# ── Full Intake Agent output ──────────────────────────────────────────
class IntakeResult(BaseModel):
    """
    Complete output of the Intake Agent.
    This is the structured payload passed downstream to Research / Analysis agents.
    """
    metadata: DocumentMetadata
    clauses: list[Clause]
    raw_text: str                           = Field(description="Full extracted text for fallback / search")
    processing_notes: list[str]             = Field(default_factory=list, description="Warnings or info from processing")

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    def clause_summary_table(self) -> list[dict]:
        """Quick summary for display / logging."""
        return [
            {
                "index": c.index,
                "heading": c.heading[:50],
                "type": c.clause_type.value,
                "risk": c.risk_level.value,
                "summary": c.summary[:80],
            }
            for c in self.clauses
        ]
