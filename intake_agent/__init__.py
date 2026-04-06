"""
Intake Agent — Agentic Lawyer Pipeline Stage 1
===============================================
Extracts, segments, and classifies legal document clauses.
"""

from .agent import IntakeAgent
from .models import IntakeResult, Clause, ClauseType, RiskLevel, DocumentMetadata

__all__ = [
    "IntakeAgent",
    "IntakeResult",
    "Clause",
    "ClauseType",
    "RiskLevel",
    "DocumentMetadata",
]
