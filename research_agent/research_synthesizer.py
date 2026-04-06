"""
Research Synthesizer
====================
Uses Claude to:
  1. Generate smart search queries from clause text + type
  2. Synthesize retrieved legal sources into actionable research
  3. Assess enforceability and provide recommendations

This is the "brain" of the Research Agent — it turns raw retrieval
results into structured legal analysis.
"""

from __future__ import annotations
import json
import time
import anthropic
from .models import LegalSource, ClauseResearch


# ── Query generation prompt ───────────────────────────────────────────

QUERY_GEN_PROMPT = """You are a legal research assistant. Given a contract clause, generate 
search queries to find relevant legal authorities.

You will receive:
- The clause text
- The clause type (e.g., "non_compete", "indemnification")
- The jurisdiction (if known)
- The risk level

Generate 3-5 diverse search queries that would help retrieve relevant:
- Statutes and codes
- Case law principles
- Regulatory requirements
- Legal standards and tests

Make queries specific to the jurisdiction when known. Vary your angles:
- One query for the core legal standard
- One for enforceability criteria
- One for common limitations or defenses
- One for recent developments or trends

Respond with a JSON array of query strings. No markdown, no explanation.
Example: ["non-compete enforceability Delaware", "restrictive covenant reasonable scope duration"]"""


# ── Research synthesis prompt ─────────────────────────────────────────

SYNTHESIS_PROMPT = """You are an expert legal research analyst. Synthesize the retrieved legal 
sources into a structured analysis of a specific contract clause.

You will receive:
- The clause text from the contract
- The clause type and risk level from initial classification
- Retrieved legal sources (statutes, case law, regulations)

Produce a JSON object with these fields:

{
  "legal_context": "<2-3 paragraph overview of the legal framework governing this type of clause>",
  "applicable_law": ["<specific statute/regulation 1>", "<specific statute/regulation 2>", ...],
  "risk_analysis": "<1-2 paragraphs analyzing how this specific clause measures against legal standards and common market practices>",
  "recommendations": ["<specific actionable recommendation 1>", "<recommendation 2>", ...],
  "enforceability_notes": "<1 paragraph on likely enforceability given the jurisdiction and clause terms>"
}

Guidelines:
- Be specific — cite the actual legal provisions from the retrieved sources
- Risk analysis should compare the clause against what courts typically enforce
- Recommendations should be concrete and negotiation-ready
- If sources are insufficient, say so and note what additional research is needed
- Use plain language accessible to non-lawyers while maintaining legal accuracy

Respond ONLY with the JSON object. No markdown fences."""


class ResearchSynthesizer:
    """
    Generates search queries and synthesizes legal research using Claude.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
    ):
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.max_retries = max_retries

    # ── Query generation ──────────────────────────────────────────────

    def generate_queries(
        self,
        clause_text: str,
        clause_type: str,
        jurisdiction: str = "",
        risk_level: str = "",
    ) -> list[str]:
        """
        Generate diverse search queries for a given clause.

        Returns 3-5 search query strings optimized for vector similarity search.
        """
        user_message = (
            f"Clause type: {clause_type}\n"
            f"Risk level: {risk_level}\n"
            f"Jurisdiction: {jurisdiction or 'Not specified'}\n\n"
            f"Clause text:\n{clause_text[:1500]}"
        )

        response = self._call_api(
            system=QUERY_GEN_PROMPT,
            user_message=user_message,
            max_tokens=512,
        )

        try:
            queries = json.loads(response)
            if isinstance(queries, list):
                return [q for q in queries if isinstance(q, str)]
        except json.JSONDecodeError:
            pass

        # Fallback: generate basic queries from clause type
        base_queries = [
            f"{clause_type.replace('_', ' ')} legal requirements",
            f"{clause_type.replace('_', ' ')} enforceability standards",
        ]
        if jurisdiction:
            base_queries.append(f"{clause_type.replace('_', ' ')} {jurisdiction} law")
        return base_queries

    # ── Research synthesis ────────────────────────────────────────────

    def synthesize(
        self,
        clause_text: str,
        clause_type: str,
        risk_level: str,
        sources: list[LegalSource],
        jurisdiction: str = "",
    ) -> dict:
        """
        Synthesize retrieved sources into structured legal research.

        Returns a dict with legal_context, applicable_law, risk_analysis,
        recommendations, and enforceability_notes.
        """
        # Format sources for the prompt
        sources_text = self._format_sources(sources)

        user_message = (
            f"CLAUSE TYPE: {clause_type}\n"
            f"RISK LEVEL: {risk_level}\n"
            f"JURISDICTION: {jurisdiction or 'Not specified'}\n\n"
            f"CLAUSE TEXT:\n{clause_text[:2000]}\n\n"
            f"RETRIEVED LEGAL SOURCES:\n{sources_text}"
        )

        response = self._call_api(
            system=SYNTHESIS_PROMPT,
            user_message=user_message,
            max_tokens=2048,
        )

        try:
            result = json.loads(response)
            return {
                "legal_context": result.get("legal_context", ""),
                "applicable_law": result.get("applicable_law", []),
                "risk_analysis": result.get("risk_analysis", ""),
                "recommendations": result.get("recommendations", []),
                "enforceability_notes": result.get("enforceability_notes", ""),
            }
        except json.JSONDecodeError:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(response[start : end + 1])
                except json.JSONDecodeError:
                    pass

            return {
                "legal_context": "Synthesis failed — raw sources are available for manual review.",
                "applicable_law": [],
                "risk_analysis": "",
                "recommendations": ["Manual review of retrieved sources recommended"],
                "enforceability_notes": "",
            }

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _format_sources(sources: list[LegalSource], max_sources: int = 8) -> str:
        """Format retrieved sources for the synthesis prompt."""
        if not sources:
            return "(No relevant sources were retrieved from the legal corpus.)"

        formatted: list[str] = []
        for i, source in enumerate(sources[:max_sources]):
            formatted.append(
                f"--- SOURCE {i + 1} ---\n"
                f"Type: {source.source_type.value}\n"
                f"Title: {source.title}\n"
                f"Jurisdiction: {source.jurisdiction}\n"
                f"Relevance: {1 - source.relevance_score:.2f}\n"
                f"Text:\n{source.text[:1200]}\n"
            )
        return "\n".join(formatted)

    def _call_api(self, system: str, user_message: str, max_tokens: int = 2048) -> str:
        """Make an API call with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user_message}],
                )
                return response.content[0].text

            except anthropic.RateLimitError:
                wait = 2 ** attempt * 5
                print(f"  ⏳ Rate limited. Waiting {wait}s...")
                time.sleep(wait)

            except anthropic.APIError as e:
                if attempt < self.max_retries - 1:
                    print(f"  ⚠️  API error (attempt {attempt + 1}): {e}")
                    time.sleep(2)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")
