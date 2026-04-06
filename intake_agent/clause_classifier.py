"""
Clause Classifier (LLM-Powered)
================================
Uses Claude to classify each segmented clause:
  - Assign a ClauseType
  - Assess risk level from the reviewing party's perspective
  - Generate a plain-language summary
  - Extract key legal terms
  - Flag potential issues or concerns

Design decisions:
  - Batch clauses into a single API call (cheaper, faster) when document is small
  - Fall back to per-clause calls for large documents (avoids context overflow)
  - Structured JSON output via system prompt engineering
  - Retry logic for transient API failures
"""

from __future__ import annotations
import json
import time
import anthropic
from .models import Clause, ClauseType, RiskLevel, DocumentMetadata


# ── Classification prompt ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a legal document analysis system. You classify contract clauses 
and assess their risk profile.

You will receive a list of clauses extracted from a legal document. For each clause, respond 
with a JSON array where each element has these fields:

{
  "index": <int, the clause index provided>,
  "clause_type": <string, one of the valid types listed below>,
  "risk_level": <"low" | "medium" | "high" | "none">,
  "summary": <string, 1-2 sentence plain-language summary of what this clause does>,
  "key_terms": <list of strings, important legal terms or defined terms found>,
  "flags": <list of strings, potential issues, missing protections, or concerns>
}

Valid clause_type values:
  definitions, confidentiality, non_disclosure, obligations, indemnification,
  termination, governing_law, dispute_resolution, liability, intellectual_property,
  representations, non_compete, non_solicitation, payment_terms, scope_of_work,
  force_majeure, assignment, severability, entire_agreement, amendments,
  notices, preamble, signature_block, miscellaneous, unknown

Risk assessment guidelines:
  - HIGH: One-sided indemnification, unlimited liability, overly broad non-competes,
    automatic renewal with no exit, unilateral amendment rights, broad IP assignment
  - MEDIUM: Standard but potentially negotiable terms, moderately broad restrictions,
    short cure periods, ambiguous language
  - LOW: Balanced/mutual obligations, standard protective language, clear and fair terms
  - NONE: Preamble, definitions, signature blocks, procedural clauses

Respond ONLY with the JSON array. No markdown fences, no commentary."""


METADATA_PROMPT = """You are a legal document analysis system. Given the full text of a legal 
document, extract high-level metadata.

Respond with a JSON object:
{
  "document_type": <string: "nda", "freelance_contract", "employment_agreement", 
                    "lease", "service_agreement", "partnership_agreement", 
                    "software_license", "consulting_agreement", "other">,
  "parties": <list of strings: the named parties in the agreement>,
  "effective_date": <string or null: the effective date if stated>,
  "jurisdiction": <string or null: governing law jurisdiction if stated>
}

Respond ONLY with the JSON object. No markdown, no explanation."""


class ClauseClassifier:
    """
    Classifies clauses using the Anthropic API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        batch_size: int = 15,
    ):
        """
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Which Claude model to use
            max_retries: Retry count for transient failures
            batch_size: Max clauses per API call
        """
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.max_retries = max_retries
        self.batch_size = batch_size

    # ── Main classification entry point ───────────────────────────────

    def classify_clauses(self, clauses: list[Clause]) -> list[Clause]:
        """
        Classify all clauses. Batches them if there are many.
        Returns the same Clause objects with classification fields populated.
        """
        if not clauses:
            return clauses

        # Process in batches
        all_results: list[dict] = []
        for i in range(0, len(clauses), self.batch_size):
            batch = clauses[i : i + self.batch_size]
            results = self._classify_batch(batch)
            all_results.extend(results)

        # Merge classification results back into Clause objects
        results_by_index = {r["index"]: r for r in all_results}

        for clause in clauses:
            if clause.index in results_by_index:
                r = results_by_index[clause.index]
                clause.clause_type = self._parse_clause_type(r.get("clause_type", "unknown"))
                clause.risk_level = self._parse_risk_level(r.get("risk_level", "none"))
                clause.summary = r.get("summary", "")
                clause.key_terms = r.get("key_terms", [])
                clause.flags = r.get("flags", [])

        return clauses

    def extract_metadata(self, full_text: str, filename: str, total_pages: int) -> DocumentMetadata:
        """
        Extract document-level metadata using LLM.
        """
        # Truncate to first ~3000 chars for metadata (preamble + first sections)
        truncated = full_text[:3000]

        response = self._call_api(
            system=METADATA_PROMPT,
            user_message=f"Document filename: {filename}\n\nDocument text (first section):\n{truncated}",
        )

        try:
            data = json.loads(response)
            return DocumentMetadata(
                filename=filename,
                total_pages=total_pages,
                document_type=data.get("document_type", "unknown"),
                parties=data.get("parties", []),
                effective_date=data.get("effective_date"),
                jurisdiction=data.get("jurisdiction"),
            )
        except (json.JSONDecodeError, KeyError):
            return DocumentMetadata(
                filename=filename,
                total_pages=total_pages,
                processing_notes=["Metadata extraction failed — using defaults"],
            )

    # ── Batch classification ──────────────────────────────────────────

    def _classify_batch(self, clauses: list[Clause]) -> list[dict]:
        """Send a batch of clauses to the API for classification."""

        # Build the user message with clause texts
        clause_entries = []
        for clause in clauses:
            # Truncate very long clauses to save tokens
            text = clause.text[:2000] if len(clause.text) > 2000 else clause.text
            clause_entries.append(
                f"--- CLAUSE {clause.index} ---\n"
                f"Heading: {clause.heading}\n"
                f"Text:\n{text}\n"
            )

        user_message = (
            f"Classify the following {len(clauses)} clauses from a legal document.\n\n"
            + "\n".join(clause_entries)
        )

        response = self._call_api(system=SYSTEM_PROMPT, user_message=user_message)

        try:
            results = json.loads(response)
            if isinstance(results, list):
                return results
        except json.JSONDecodeError:
            # Try to extract JSON from response
            results = self._extract_json_array(response)
            if results:
                return results

        # Fallback: return empty classification for each clause
        return [{"index": c.index, "clause_type": "unknown", "risk_level": "none",
                 "summary": "Classification failed", "key_terms": [], "flags": []}
                for c in clauses]

    # ── API call with retry ───────────────────────────────────────────

    def _call_api(self, system: str, user_message: str) -> str:
        """Make an API call with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system,
                    messages=[{"role": "user", "content": user_message}],
                )
                return response.content[0].text

            except anthropic.RateLimitError:
                wait = 2 ** attempt * 5
                print(f"  ⏳ Rate limited. Waiting {wait}s before retry {attempt + 1}...")
                time.sleep(wait)

            except anthropic.APIError as e:
                if attempt < self.max_retries - 1:
                    print(f"  ⚠️  API error (attempt {attempt + 1}): {e}")
                    time.sleep(2)
                else:
                    raise

        raise RuntimeError("Max retries exceeded for API call")

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_clause_type(value: str) -> ClauseType:
        try:
            return ClauseType(value.lower().strip())
        except ValueError:
            return ClauseType.UNKNOWN

    @staticmethod
    def _parse_risk_level(value: str) -> RiskLevel:
        try:
            return RiskLevel(value.lower().strip())
        except ValueError:
            return RiskLevel.NONE

    @staticmethod
    def _extract_json_array(text: str) -> list[dict] | None:
        """Try to extract a JSON array from messy LLM output."""
        # Find the outermost [ ... ]
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return None
