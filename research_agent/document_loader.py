"""
Legal Document Loader
=====================
Loads, chunks, and prepares legal documents for vector store ingestion.

Chunking strategy for legal text:
  - Split on section boundaries first (statutes have natural sections)
  - Fall back to paragraph-level splits
  - Overlap between chunks to preserve cross-references
  - Attach metadata (source type, jurisdiction, title, section) to each chunk

Supported input formats:
  - Plain text (.txt)
  - JSON corpus files (.json) — structured collections of legal provisions
"""

from __future__ import annotations
import json
import re
import uuid
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class LegalChunk:
    """A single chunk of legal text ready for embedding."""
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class LegalDocumentLoader:
    """
    Loads and chunks legal documents for vector store ingestion.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        """
        Args:
            chunk_size: Target size in characters per chunk.
            chunk_overlap: Character overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Load from structured JSON corpus ──────────────────────────────

    def load_json_corpus(self, path: str | Path) -> list[LegalChunk]:
        """
        Load a structured JSON corpus file.

        Expected format:
        {
          "source_type": "statute",
          "jurisdiction": "Delaware",
          "title": "Delaware Code Title 6 - Commerce and Trade",
          "sections": [
            {
              "section_id": "6-2701",
              "heading": "Definitions",
              "text": "As used in this chapter..."
            }
          ]
        }
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        source_type = data.get("source_type", "unknown")
        jurisdiction = data.get("jurisdiction", "")
        title = data.get("title", path.stem)

        chunks: list[LegalChunk] = []

        for section in data.get("sections", []):
            section_text = section.get("text", "")
            section_heading = section.get("heading", "")
            section_id = section.get("section_id", "")

            if not section_text.strip():
                continue

            # Chunk long sections, keep short ones whole
            if len(section_text) <= self.chunk_size:
                chunks.append(LegalChunk(
                    chunk_id=f"{path.stem}_{section_id}_{uuid.uuid4().hex[:6]}",
                    text=section_text.strip(),
                    metadata={
                        "source_type": source_type,
                        "jurisdiction": jurisdiction,
                        "title": title,
                        "section_id": section_id,
                        "section_heading": section_heading,
                        "source_file": str(path.name),
                    },
                ))
            else:
                sub_chunks = self._chunk_text(section_text)
                for i, chunk_text in enumerate(sub_chunks):
                    chunks.append(LegalChunk(
                        chunk_id=f"{path.stem}_{section_id}_p{i}_{uuid.uuid4().hex[:6]}",
                        text=chunk_text,
                        metadata={
                            "source_type": source_type,
                            "jurisdiction": jurisdiction,
                            "title": title,
                            "section_id": section_id,
                            "section_heading": section_heading,
                            "chunk_index": i,
                            "source_file": str(path.name),
                        },
                    ))

        return chunks

    # ── Load from plain text ──────────────────────────────────────────

    def load_text_file(
        self,
        path: str | Path,
        source_type: str = "unknown",
        jurisdiction: str = "",
        title: str = "",
    ) -> list[LegalChunk]:
        """Load and chunk a plain text legal document."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        title = title or path.stem

        # Try to split on section headings first
        sections = self._split_on_sections(text)

        chunks: list[LegalChunk] = []
        for i, (heading, section_text) in enumerate(sections):
            sub_chunks = self._chunk_text(section_text) if len(section_text) > self.chunk_size else [section_text]

            for j, chunk_text in enumerate(sub_chunks):
                chunks.append(LegalChunk(
                    chunk_id=f"{path.stem}_s{i}_p{j}_{uuid.uuid4().hex[:6]}",
                    text=chunk_text.strip(),
                    metadata={
                        "source_type": source_type,
                        "jurisdiction": jurisdiction,
                        "title": title,
                        "section_heading": heading,
                        "source_file": str(path.name),
                    },
                ))

        return chunks

    # ── Load from inline data (for seeding) ───────────────────────────

    def load_inline(
        self,
        sections: list[dict],
        source_type: str = "statute",
        jurisdiction: str = "",
        title: str = "",
    ) -> list[LegalChunk]:
        """
        Load pre-structured sections directly (used by corpus seeder).

        Each section dict should have:
          - section_id: str
          - heading: str
          - text: str
        """
        chunks: list[LegalChunk] = []
        for section in sections:
            text = section.get("text", "")
            if not text.strip():
                continue

            section_id = section.get("section_id", uuid.uuid4().hex[:8])
            heading = section.get("heading", "")

            if len(text) <= self.chunk_size:
                chunks.append(LegalChunk(
                    chunk_id=f"inline_{section_id}_{uuid.uuid4().hex[:6]}",
                    text=text.strip(),
                    metadata={
                        "source_type": source_type,
                        "jurisdiction": jurisdiction,
                        "title": title,
                        "section_id": section_id,
                        "section_heading": heading,
                    },
                ))
            else:
                sub_chunks = self._chunk_text(text)
                for i, chunk_text in enumerate(sub_chunks):
                    chunks.append(LegalChunk(
                        chunk_id=f"inline_{section_id}_p{i}_{uuid.uuid4().hex[:6]}",
                        text=chunk_text,
                        metadata={
                            "source_type": source_type,
                            "jurisdiction": jurisdiction,
                            "title": title,
                            "section_id": section_id,
                            "section_heading": heading,
                            "chunk_index": i,
                        },
                    ))

        return chunks

    # ── Chunking logic ────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.

        Strategy:
          1. Split on paragraph boundaries (double newline)
          2. Accumulate paragraphs until chunk_size is reached
          3. Start next chunk with overlap from the previous
        """
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for para in paragraphs:
            if current_length + len(para) > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))

                # Calculate overlap: keep last paragraphs up to overlap size
                overlap_parts: list[str] = []
                overlap_len = 0
                for p in reversed(current_chunk):
                    if overlap_len + len(p) > self.chunk_overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_len += len(p)

                current_chunk = overlap_parts
                current_length = overlap_len

            current_chunk.append(para)
            current_length += len(para)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    @staticmethod
    def _split_on_sections(text: str) -> list[tuple[str, str]]:
        """
        Split text on section-like headings.
        Returns list of (heading, text) tuples.
        """
        pattern = re.compile(
            r"^(?:(?:Section|§|Art(?:icle)?\.?)\s+[\d\.]+[:\.\s\-]*(.+)|"
            r"([A-Z][A-Z\s\-]{3,50}))$",
            re.MULTILINE,
        )

        matches = list(pattern.finditer(text))

        if not matches:
            return [("", text)]

        sections: list[tuple[str, str]] = []

        # Preamble before first heading
        if matches[0].start() > 0:
            sections.append(("Preamble", text[: matches[0].start()].strip()))

        for i, match in enumerate(matches):
            heading = (match.group(1) or match.group(2) or "").strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            sections.append((heading, section_text))

        return sections
