"""
Evaluator Agent — Agentic Lawyer Pipeline Stage 4
===================================================
Quality gate: validates grounding, consistency, and confidence of analysis.
Triggers reflection loop for low-confidence findings.
"""

from .agent import EvaluatorAgent
from .models import EvaluationResult, ClauseEvaluation, FindingEvaluation, ConfidenceLevel

__all__ = [
    "EvaluatorAgent",
    "EvaluationResult",
    "ClauseEvaluation",
    "FindingEvaluation",
    "ConfidenceLevel",
]
