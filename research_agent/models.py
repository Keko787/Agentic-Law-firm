"""
Data models for the Research Agent pipeline.

These models define:
  LegalSource     → a single retrieved legal reference
  ClauseResearch  → research findings for one clause
  ResearchResult  → full output bundle from the Research Agent
"""

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class SourceType(str, Enum):
    """Types of legal sources the Research Agent retrieves."""
    STATUTE         = "statute"
    CASE_LAW        = "case_law"
    REGULATION      = "regulation"
    LEGAL_TREATISE  = "legal_treatise"
    RESTATEMENT     = "restatement"
    MODEL_CODE      = "model_code"
    COMMENTARY      = "commentary"
    UNKNOWN         = "unknown"


class LegalSource(BaseModel):
    """
    A single legal reference retrieved from the vector store.
    """
    source_id: str                          = Field(description="Unique ID from the vector store")
    source_type: SourceType                 = Field(default=SourceType.UNKNOWN)
    title: str                              = Field(description="Title or citation of the source")
    jurisdiction: str                       = Field(default="", description="e.g., 'Delaware', 'Federal', 'California'")
    text: str                               = Field(description="The retrieved text passage")
    relevance_score: float                  = Field(default=0.0, description="Similarity score from retrieval")
    metadata: dict                          = Field(default_factory=dict, description="Additional metadata from the corpus")


class ClauseResearch(BaseModel):
    """
    Research findings for a single clause from the intake result.
    Links back to the clause by index.
    """
    clause_index: int                       = Field(description="Index of the clause from IntakeResult")
    clause_heading: str                     = Field(description="Heading from the intake clause")
    clause_type: str                        = Field(description="ClauseType value from intake")
    risk_level: str                         = Field(description="RiskLevel value from intake")

    # Retrieved sources
    sources: list[LegalSource]              = Field(default_factory=list)

    # LLM-synthesized research
    legal_context: str                      = Field(default="", description="Synthesized legal background for this clause")
    applicable_law: list[str]               = Field(default_factory=list, description="Specific statutes/regulations that apply")
    risk_analysis: str                      = Field(default="", description="How the clause measures against legal standards")
    recommendations: list[str]              = Field(default_factory=list, description="Suggested modifications or negotiation points")
    enforceability_notes: str               = Field(default="", description="Notes on likely enforceability")

    # Search metadata
    queries_used: list[str]                 = Field(default_factory=list, description="Search queries that produced these results")


class ResearchResult(BaseModel):
    """
    Complete output of the Research Agent.
    Passed downstream to the Analysis Agent.
    """
    document_filename: str
    document_type: str
    jurisdiction: str                       = Field(default="")
    clause_research: list[ClauseResearch]   = Field(default_factory=list)
    processing_notes: list[str]             = Field(default_factory=list)

    # Corpus stats
    corpus_size: int                        = Field(default=0, description="Number of documents in the vector store")
    total_sources_retrieved: int            = Field(default=0)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    def high_risk_research(self) -> list[ClauseResearch]:
        """Return only research for high-risk clauses."""
        return [cr for cr in self.clause_research if cr.risk_level == "high"]
