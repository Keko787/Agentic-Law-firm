"""
Analysis Agent — Agentic Lawyer Pipeline Stage 3
==================================================
Merges intake classification with legal research to produce
grounded, actionable analysis with findings, scoring, and recommendations.
"""

from .agent import AnalysisAgent
from .models import (
    AnalysisResult, ClauseAnalysis, Finding, Severity, IssueCategory,
    MissingClause, DocumentScore,
)

__all__ = [
    "AnalysisAgent",
    "AnalysisResult",
    "ClauseAnalysis",
    "Finding",
    "Severity",
    "IssueCategory",
    "MissingClause",
    "DocumentScore",
]
