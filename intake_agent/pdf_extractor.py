"""
PDF Text Extractor
==================
Extracts raw text from PDF documents using pdfplumber.

Why pdfplumber over PyPDF2?
  - Better handling of complex layouts, tables, headers/footers
  - Preserves reading order more reliably
  - Handles multi-column layouts

Usage:
    extractor = PDFExtractor()
    pages = extractor.extract("contract.pdf")
    full_text = extractor.extract_full_text("contract.pdf")
"""

from __future__ import annotations
import pdfplumber
from pathlib import Path
from .models import RawPage


class PDFExtractor:
    """Extracts and cleans text from PDF files."""

    def __init__(self, deskew: bool = False):
        self.deskew = deskew

    def extract(self, pdf_path: str | Path) -> list[RawPage]:
        """
        Extract text from each page of a PDF.

        Returns a list of RawPage objects with cleaned text.
        Raises FileNotFoundError if the path doesn't exist.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages: list[RawPage] = []

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                raw = page.extract_text() or ""
                cleaned = self._clean_page_text(raw, page_num=i + 1)
                pages.append(RawPage(page_number=i + 1, text=cleaned))

        return pages

    def extract_full_text(self, pdf_path: str | Path) -> str:
        """Extract and join all pages into a single string."""
        pages = self.extract(pdf_path)
        return "\n\n".join(p.text for p in pages if p.text.strip())

    def get_page_count(self, pdf_path: str | Path) -> int:
        """Return total number of pages."""
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)

    # ── Internal helpers ──────────────────────────────────────────────

    def _clean_page_text(self, text: str, page_num: int) -> str:
        """
        Clean extracted text:
          - Strip excessive whitespace
          - Remove common headers/footers (page numbers, "Confidential", etc.)
          - Normalize line breaks
        """
        lines = text.split("\n")
        cleaned_lines: list[str] = []

        for line in lines:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                cleaned_lines.append("")
                continue

            # Skip standalone page numbers
            if stripped.isdigit() or stripped.lower().startswith("page "):
                continue

            # Skip common footer patterns
            if self._is_boilerplate_footer(stripped):
                continue

            cleaned_lines.append(stripped)

        # Collapse runs of 3+ blank lines into 2
        result: list[str] = []
        blank_count = 0
        for line in cleaned_lines:
            if line == "":
                blank_count += 1
                if blank_count <= 2:
                    result.append(line)
            else:
                blank_count = 0
                result.append(line)

        return "\n".join(result).strip()

    @staticmethod
    def _is_boilerplate_footer(line: str) -> bool:
        """Detect common contract footer boilerplate."""
        lower = line.lower()
        patterns = [
            "confidential",
            "all rights reserved",
            "proprietary and confidential",
            "draft – for discussion purposes only",
        ]
        # Only match if the line is SHORT and matches a pattern
        # (long lines containing "confidential" are likely real content)
        if len(lower) < 60:
            return any(lower.strip() == p for p in patterns)
        return False
