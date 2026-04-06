"""
Intake Agent — Main Orchestrator
=================================
The Intake Agent is the entry point of the Agentic Lawyer pipeline.

Responsibilities:
  1. Accept a PDF legal document
  2. Extract raw text (PDFExtractor)
  3. Segment into clauses (ClauseSegmenter)
  4. Classify each clause via LLM (ClauseClassifier)
  5. Extract document metadata (ClauseClassifier)
  6. Produce a structured IntakeResult for downstream agents

Usage:
    agent = IntakeAgent(api_key="sk-ant-...")
    result = agent.process("contract.pdf")
    result.save("output/intake_result.json")
"""

from __future__ import annotations
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .pdf_extractor import PDFExtractor
from .clause_segmenter import ClauseSegmenter
from .clause_classifier import ClauseClassifier
from .models import IntakeResult


console = Console()


class IntakeAgent:
    """
    Orchestrates the full intake pipeline: extract → segment → classify.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        verbose: bool = True,
    ):
        self.extractor = PDFExtractor()
        self.segmenter = ClauseSegmenter(min_clause_length=50)
        self.classifier = ClauseClassifier(api_key=api_key, model=model)
        self.verbose = verbose

    def process(self, pdf_path: str | Path) -> IntakeResult:
        """
        Run the full intake pipeline on a PDF document.

        Args:
            pdf_path: Path to the PDF file to process.

        Returns:
            IntakeResult with metadata, classified clauses, and raw text.
        """
        pdf_path = Path(pdf_path)
        notes: list[str] = []

        # ── Step 1: Extract text ──────────────────────────────────────
        self._log("📄 Step 1/4 — Extracting text from PDF...")
        pages = self.extractor.extract(pdf_path)
        full_text = "\n\n".join(p.text for p in pages if p.text.strip())
        total_pages = len(pages)
        self._log(f"   Extracted {total_pages} pages, {len(full_text):,} characters")

        if not full_text.strip():
            notes.append("WARNING: No text extracted — PDF may be scanned/image-based")
            self._log("   ⚠️  No text found. Consider OCR preprocessing.", style="yellow")

        # ── Step 2: Segment into clauses ──────────────────────────────
        self._log("✂️  Step 2/4 — Segmenting into clauses...")
        clauses = self.segmenter.segment(pages)
        self._log(f"   Found {len(clauses)} clauses")

        if len(clauses) == 1:
            notes.append("INFO: Only one clause detected — document may lack clear section headings")

        # ── Step 3: Classify clauses via LLM ──────────────────────────
        self._log("🧠 Step 3/4 — Classifying clauses (LLM)...")
        clauses = self.classifier.classify_clauses(clauses)
        self._log(f"   Classified {len(clauses)} clauses")

        # ── Step 4: Extract document metadata ─────────────────────────
        self._log("📋 Step 4/4 — Extracting document metadata...")
        metadata = self.classifier.extract_metadata(full_text, pdf_path.name, total_pages)
        self._log(f"   Type: {metadata.document_type}")
        self._log(f"   Parties: {', '.join(metadata.parties) if metadata.parties else 'not detected'}")

        # ── Build result ──────────────────────────────────────────────
        result = IntakeResult(
            metadata=metadata,
            clauses=clauses,
            raw_text=full_text,
            processing_notes=notes,
        )

        if self.verbose:
            self._print_summary(result)

        return result

    # ── Display helpers ───────────────────────────────────────────────

    def _log(self, message: str, style: str = "cyan") -> None:
        if self.verbose:
            console.print(message, style=style)

    def _print_summary(self, result: IntakeResult) -> None:
        """Print a formatted summary table to the console."""
        console.print()
        console.print(
            Panel(
                f"[bold]{result.metadata.filename}[/bold]\n"
                f"Type: {result.metadata.document_type}  |  "
                f"Pages: {result.metadata.total_pages}  |  "
                f"Clauses: {len(result.clauses)}",
                title="📑 Intake Agent — Results",
                border_style="green",
            )
        )

        # Clause table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", width=4, justify="center")
        table.add_column("Heading", width=25)
        table.add_column("Type", width=20)
        table.add_column("Risk", width=8, justify="center")
        table.add_column("Summary", width=50)
        table.add_column("Flags", width=30)

        risk_colors = {"high": "red", "medium": "yellow", "low": "green", "none": "dim"}

        for clause in result.clauses:
            risk_style = risk_colors.get(clause.risk_level.value, "white")
            flags_str = "; ".join(clause.flags[:2]) if clause.flags else "—"

            table.add_row(
                str(clause.index),
                clause.heading[:25],
                clause.clause_type.value,
                f"[{risk_style}]{clause.risk_level.value.upper()}[/{risk_style}]",
                clause.summary[:50] + "..." if len(clause.summary) > 50 else clause.summary,
                flags_str[:30],
            )

        console.print(table)

        # Risk summary
        high_risk = [c for c in result.clauses if c.risk_level.value == "high"]
        if high_risk:
            console.print(
                f"\n⚠️  [bold red]{len(high_risk)} HIGH-RISK clause(s) detected:[/bold red]"
            )
            for c in high_risk:
                console.print(f"   • [red]{c.heading}[/red]: {c.summary}")
