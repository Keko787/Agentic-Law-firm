"""
Output Agent — Agentic Lawyer Pipeline Stage 5
================================================
Generates the final plain-language report with executive summary,
clause findings, negotiation playbook, and confidence disclosure.
"""

from .agent import OutputAgent
from .models import FinalReport, ReportSection

__all__ = ["OutputAgent", "FinalReport", "ReportSection"]
