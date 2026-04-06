"""
Clause Segmenter
================
Splits raw legal document text into individual clauses.

Strategy:
  1. Detect section headings using regex patterns common in contracts
  2. Split text at heading boundaries
  3. Handle edge cases: nested sub-sections, run-on paragraphs, preambles
  4. Assign page numbers based on character offsets

This is a RULE-BASED first pass. The LLM classifier refines boundaries
if needed, but getting 80% of segmentation right here saves tokens and
makes classification more reliable.
"""

from __future__ import annotations
import re
from .models import Clause, RawPage


# ── Heading patterns (ordered by specificity) ────────────────────────
# These match common legal document section numbering styles:
#   "1. Definitions"
#   "ARTICLE II - CONFIDENTIALITY"
#   "Section 3.2  Obligations"
#   "2.1 Scope of Work"
#   "A. Representations"
#   "EXHIBIT A"

HEADING_PATTERNS = [
    # "ARTICLE I - Title" or "ARTICLE 1: Title"
    re.compile(
        r"^(?:ARTICLE|Art\.?)\s+[IVXLCDM\d]+[\s\.\-:]+\s*(.+)",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "Section 1.2 - Title" or "SECTION 3: Title"
    re.compile(
        r"^(?:SECTION|Sec\.?)\s+[\d\.]+[\s\.\-:]+\s*(.+)",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "1. Title" or "12. Title" (numbered sections)
    re.compile(
        r"^(\d{1,2})\.\s+([A-Z][A-Za-z\s\-,/&]{2,60})$",
        re.MULTILINE,
    ),
    # "1.1 Title" or "3.2.1 Title" (decimal sub-sections)
    re.compile(
        r"^(\d{1,2}(?:\.\d{1,2})+)\s+([A-Z][A-Za-z\s\-,/&]{2,60})$",
        re.MULTILINE,
    ),
    # "A. Title" or "(a) Title" (lettered sections)
    re.compile(
        r"^(?:\(?([A-Z])\)?[\.\)]\s+)([A-Z][A-Za-z\s\-,/&]{2,60})$",
        re.MULTILINE,
    ),
    # ALL-CAPS headings on their own line (e.g. "CONFIDENTIALITY")
    re.compile(
        r"^([A-Z][A-Z\s\-,/&]{4,60})$",
        re.MULTILINE,
    ),
    # "EXHIBIT A" / "SCHEDULE 1"
    re.compile(
        r"^(?:EXHIBIT|SCHEDULE|APPENDIX|ANNEX)\s+[A-Z\d]+",
        re.IGNORECASE | re.MULTILINE,
    ),
]


class ClauseSegmenter:
    """
    Segments legal document text into clauses based on heading detection.
    """

    def __init__(self, min_clause_length: int = 50):
        """
        Args:
            min_clause_length: Minimum character count for a clause to be
                               kept as standalone. Shorter chunks get merged
                               into the previous clause.
        """
        self.min_clause_length = min_clause_length

    def segment(self, pages: list[RawPage]) -> list[Clause]:
        """
        Main entry point: takes extracted pages, returns segmented clauses.

        Steps:
          1. Join pages into full text (tracking page boundaries)
          2. Find all heading positions
          3. Split text at headings
          4. Build Clause objects with page-number attribution
        """
        full_text, page_boundaries = self._join_pages(pages)
        headings = self._find_headings(full_text)

        # If no headings found, treat the whole document as one clause
        if not headings:
            return [
                Clause(
                    index=0,
                    heading="Full Document",
                    text=full_text.strip(),
                    source_pages=list(range(1, len(pages) + 1)),
                )
            ]

        clauses = self._split_at_headings(full_text, headings, page_boundaries)
        clauses = self._merge_short_clauses(clauses)
        return clauses

    # ── Heading detection ─────────────────────────────────────────────

    def _find_headings(self, text: str) -> list[dict]:
        """
        Find all section headings and their character positions.

        Returns a list of {position, heading_text, match_text} dicts,
        sorted by position.
        """
        found: list[dict] = []
        seen_positions: set[int] = set()

        for pattern in HEADING_PATTERNS:
            for match in pattern.finditer(text):
                pos = match.start()
                # Avoid duplicate detections at the same position
                if any(abs(pos - s) < 10 for s in seen_positions):
                    continue

                heading_text = match.group(0).strip()

                # Filter out false positives
                if self._is_false_positive(heading_text, text, pos):
                    continue

                seen_positions.add(pos)
                found.append({
                    "position": pos,
                    "heading": self._normalize_heading(heading_text),
                    "raw": heading_text,
                })

        # Sort by position in document
        found.sort(key=lambda x: x["position"])
        return found

    def _is_false_positive(self, heading: str, text: str, pos: int) -> bool:
        """Filter out lines that look like headings but aren't."""
        h_lower = heading.lower().strip()

        # Too short
        if len(h_lower) < 3:
            return True

        # Common false positives for all-caps pattern
        false_caps = {
            "and", "the", "or", "in", "to", "for", "of", "by",
            "whereas", "now therefore", "witnesseth", "recitals",
        }
        if h_lower in false_caps:
            return True

        # If the "heading" is actually mid-sentence (preceded by comma or lowercase)
        if pos > 0:
            before = text[max(0, pos - 5):pos].strip()
            if before and before[-1] in (",", ";", "and"[-1]):
                return True

        return False

    @staticmethod
    def _normalize_heading(heading: str) -> str:
        """Clean up heading text for display."""
        # Remove leading numbers, dots, dashes
        cleaned = re.sub(r"^[\d\.\-\(\):]+\s*", "", heading)
        # Remove "ARTICLE", "SECTION" prefixes
        cleaned = re.sub(r"^(?:ARTICLE|SECTION|Art|Sec)\.?\s*[IVXLCDM\d]*[\s\.\-:]*", "", cleaned, flags=re.IGNORECASE)
        # Title case if ALL CAPS
        if cleaned.isupper() and len(cleaned) > 3:
            cleaned = cleaned.title()
        return cleaned.strip() or heading.strip()

    # ── Text splitting ────────────────────────────────────────────────

    @staticmethod
    def _join_pages(pages: list[RawPage]) -> tuple[str, list[tuple[int, int]]]:
        """
        Join page texts into one string, tracking page boundaries.

        Returns:
            (full_text, boundaries) where boundaries is a list of
            (start_char, page_number) tuples.
        """
        full_text = ""
        boundaries: list[tuple[int, int]] = []

        for page in pages:
            boundaries.append((len(full_text), page.page_number))
            full_text += page.text + "\n\n"

        return full_text, boundaries

    def _split_at_headings(
        self,
        text: str,
        headings: list[dict],
        page_boundaries: list[tuple[int, int]],
    ) -> list[Clause]:
        """Split the full text at heading positions into Clause objects."""
        clauses: list[Clause] = []

        # Handle preamble (text before first heading)
        if headings and headings[0]["position"] > 0:
            preamble_text = text[: headings[0]["position"]].strip()
            if len(preamble_text) > self.min_clause_length:
                clauses.append(
                    Clause(
                        index=0,
                        heading="Preamble",
                        text=preamble_text,
                        source_pages=self._get_pages_for_span(
                            0, headings[0]["position"], page_boundaries
                        ),
                    )
                )

        # Split between headings
        for i, heading in enumerate(headings):
            start = heading["position"]
            end = headings[i + 1]["position"] if i + 1 < len(headings) else len(text)
            clause_text = text[start:end].strip()

            clauses.append(
                Clause(
                    index=len(clauses),
                    heading=heading["heading"],
                    text=clause_text,
                    source_pages=self._get_pages_for_span(start, end, page_boundaries),
                )
            )

        return clauses

    def _merge_short_clauses(self, clauses: list[Clause]) -> list[Clause]:
        """Merge clauses shorter than min_clause_length into their predecessor."""
        if len(clauses) <= 1:
            return clauses

        merged: list[Clause] = [clauses[0]]
        for clause in clauses[1:]:
            if len(clause.text) < self.min_clause_length and merged:
                # Append to previous clause
                prev = merged[-1]
                prev.text += "\n\n" + clause.text
                prev.source_pages = sorted(set(prev.source_pages + clause.source_pages))
            else:
                clause.index = len(merged)
                merged.append(clause)

        return merged

    @staticmethod
    def _get_pages_for_span(
        start: int, end: int, boundaries: list[tuple[int, int]]
    ) -> list[int]:
        """Determine which PDF pages a character span covers."""
        pages: list[int] = []
        for i, (bstart, page_num) in enumerate(boundaries):
            bend = boundaries[i + 1][0] if i + 1 < len(boundaries) else float("inf")
            if bstart < end and bend > start:
                pages.append(page_num)
        return pages
