# Intake Agent (Stage 1)

The Intake Agent is the entry point of the Agentic Lawyer pipeline. It takes a raw PDF legal document and produces a structured, classified breakdown of clauses that all downstream agents consume.

## What It Does

```
PDF Document
    |
    v
+-----------+     +----------------+     +---------------+     +------------+
| PDF       | --> | Clause         | --> | LLM           | --> | Metadata   |
| Extractor |     | Segmenter      |     | Classifier    |     | Extraction |
| pdfplumber|     | regex patterns |     | Claude API    |     | Claude API |
+-----------+     +----------------+     +---------------+     +------------+
    |                                                                |
    v                                                                v
IntakeResult (metadata + classified clauses + raw text)
```

## Components

### PDFExtractor (`pdf_extractor.py`)

Extracts raw text from PDF files using `pdfplumber` (chosen over PyPDF2 for better reading-order preservation and layout handling).

- Extracts text page-by-page with page-number tracking
- Cleans boilerplate: strips standalone page numbers, "Confidential" markers, "All rights reserved"
- Collapses excessive blank lines, normalizes whitespace
- Returns a list of `RawPage` objects

### ClauseSegmenter (`clause_segmenter.py`)

Rule-based heading detection that splits the document into logical clauses. Uses 7 regex patterns ordered by specificity:

1. **ARTICLE format** — `ARTICLE II - CONFIDENTIALITY`
2. **SECTION format** — `Section 3.2 - Title`
3. **Numbered sections** — `1. Definitions`
4. **Decimal sub-sections** — `3.2.1 Title`
5. **Lettered sections** — `A. Representations`
6. **ALL-CAPS headings** — `GOVERNING LAW`
7. **Exhibit/Schedule markers** — `EXHIBIT A`

Also handles:
- **Preamble detection** — text before the first heading is captured as "Preamble"
- **Short-clause merging** — clauses shorter than 50 characters are merged into their predecessor
- **Page-span attribution** — each clause knows which PDF pages it spans
- **False positive filtering** — rejects mid-sentence matches and common false positives

This rule-based approach handles ~80% of contracts without spending LLM tokens.

### ClauseClassifier (`clause_classifier.py`)

LLM-powered classification via the Claude API. Makes two API calls:

**Call 1 — Clause classification (batched):**
For each clause, returns:
- `clause_type` — one of 25 legal clause categories
- `risk_level` — `high`, `medium`, `low`, or `none`
- `summary` — 1-2 sentence plain-language summary
- `key_terms` — important legal terms found
- `flags` — potential issues or concerns

Batches up to 15 clauses per API call. Falls back to per-clause calls for large documents. Truncates clauses to 2,000 characters to save tokens.

**Call 2 — Metadata extraction:**
Runs on the first ~3,000 characters (preamble + first sections) to extract:
- Document type (NDA, service agreement, lease, etc.)
- Named parties
- Effective date
- Governing law jurisdiction

Both calls include retry logic with exponential backoff for rate limits and transient API errors.

## Risk Classification

| Level | Criteria |
|-------|----------|
| **HIGH** | One-sided indemnification, unlimited liability, overly broad non-competes, unilateral amendment rights, broad IP assignment |
| **MEDIUM** | Standard but negotiable terms, moderately broad restrictions, short cure periods, ambiguous language |
| **LOW** | Balanced/mutual obligations, standard protective language, clear and fair terms |
| **NONE** | Preamble, definitions, signature blocks, procedural clauses |

## 25 Supported Clause Types

`definitions`, `confidentiality`, `non_disclosure`, `obligations`, `indemnification`, `termination`, `governing_law`, `dispute_resolution`, `liability`, `intellectual_property`, `representations`, `non_compete`, `non_solicitation`, `payment_terms`, `scope_of_work`, `force_majeure`, `assignment`, `severability`, `entire_agreement`, `amendments`, `notices`, `preamble`, `signature_block`, `miscellaneous`, `unknown`

## Data Models (`models.py`)

| Model | Description |
|-------|-------------|
| `ClauseType` | Enum of 25 legal clause categories |
| `RiskLevel` | Enum: `low`, `medium`, `high`, `none` |
| `RawPage` | Page number + extracted text |
| `Clause` | Index, heading, text, type, risk, summary, key terms, flags, source pages |
| `DocumentMetadata` | Filename, total pages, document type, parties, effective date, jurisdiction |
| `IntakeResult` | Metadata + list of clauses + raw text + processing notes |

## Usage

```python
from intake_agent import IntakeAgent

agent = IntakeAgent(api_key="sk-ant-...")
result = agent.process("contract.pdf")
result.save("output/intake_result.json")

# Access classified clauses
for clause in result.clauses:
    print(f"{clause.heading}: {clause.clause_type.value} ({clause.risk_level.value})")

# Get high-risk clauses
high_risk = [c for c in result.clauses if c.risk_level.value == "high"]
```

## Output

The `IntakeResult` is serialized as JSON and passed to the Research Agent (Stage 2). It contains everything downstream agents need:
- Document metadata (type, parties, jurisdiction) for jurisdiction-aware research
- Typed, risk-classified clauses for prioritized analysis
- Raw text as a fallback for edge cases
- Processing notes for debugging

## Files

```
intake_agent/
├── __init__.py
├── agent.py              # IntakeAgent orchestrator (process method)
├── models.py             # Pydantic data models
├── pdf_extractor.py      # pdfplumber text extraction + cleaning
├── clause_segmenter.py   # Rule-based heading detection (7 regex patterns)
└── clause_classifier.py  # LLM classification + metadata extraction
```
