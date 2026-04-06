# Agentic Lawyer AI — Full Implementation Guide

This document captures the complete implementation strategy, architecture decisions, current state, and next steps for the Agentic Lawyer multi-agent legal document review system. Transfer this to Claude Code as the authoritative project reference.

---

## Project overview

The Agentic Lawyer is a multi-agent AI system that autonomously reviews legal documents, retrieves applicable law, flags risks, and generates plain-language reports. It uses a 5-agent pipeline orchestrated in a DAG structure, with RAG grounding and a reflection loop for quality assurance.

**Team:** Kevin Kostage, Ping Liu, Adib Bazgir, Mohsen Rezaei, Saiteja Labba

**Core thesis:** Legal document review encompasses diverse specialized tasks more efficiently performed by dedicated agents than a single LLM. Enhancements in memory, tool use, and environmental interaction are critical for deep legal reasoning.

---

## Architecture: 5-agent pipeline

```
PDF Document
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: INTAKE AGENT                                       │
│   PDF extraction → clause segmentation → LLM classification │
│   Output: IntakeResult (metadata + typed clauses + flags)   │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: RESEARCH AGENT                                     │
│   Query generation → RAG retrieval → legal synthesis        │
│   Output: ResearchResult (sources + applicable law + recs)  │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: ANALYSIS AGENT                                     │
│   Merge intake + research → deep findings → document score  │
│   Output: AnalysisResult (findings + missing + score)       │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: EVALUATOR AGENT                                    │
│   Grounding check → confidence scoring → reflection loop    │
│   Output: EvaluationResult (confidence + hallucination flags│
│   ↻ Re-analysis trigger if low confidence                   │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 5: OUTPUT AGENT                                       │
│   Executive summary → report sections → negotiation playbook│
│   Output: FinalReport (JSON + Markdown)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Project structure

```
agentic-lawyer/
├── main.py                          # CLI entry point — orchestrates all 5 stages
├── requirements.txt                 # anthropic, pdfplumber, pydantic, rich, chromadb
├── README.md                        # Project documentation
├── generate_sample_nda.py           # Creates test PDF with intentionally risky clauses
│
├── intake_agent/                    # STAGE 1 — Parse, segment, classify
│   ├── __init__.py
│   ├── agent.py                     # IntakeAgent orchestrator
│   ├── models.py                    # RawPage, Clause, ClauseType, RiskLevel, IntakeResult
│   ├── pdf_extractor.py             # pdfplumber text extraction + cleaning
│   ├── clause_segmenter.py          # Rule-based heading detection (7 regex patterns)
│   └── clause_classifier.py         # LLM classification via Claude API
│
├── research_agent/                  # STAGE 2 — RAG retrieval + synthesis
│   ├── __init__.py
│   ├── agent.py                     # ResearchAgent orchestrator
│   ├── models.py                    # LegalSource, ClauseResearch, ResearchResult
│   ├── document_loader.py           # Chunks legal texts for vector store ingestion
│   ├── vector_store.py              # ChromaDB wrapper + LocalHashEmbedding fallback
│   ├── research_synthesizer.py      # LLM query generation + source synthesis
│   └── corpus_seeder.py             # Sample legal corpus (7 sources, 15 chunks)
│
├── analysis_agent/                  # STAGE 3 — Deep analysis + scoring
│   ├── __init__.py
│   ├── agent.py                     # AnalysisAgent orchestrator
│   ├── models.py                    # Finding, ClauseAnalysis, MissingClause, DocumentScore
│   └── clause_analyzer.py           # LLM clause analysis + document-level analysis
│
├── evaluator_agent/                 # STAGE 4 — Quality gate + reflection
│   ├── __init__.py
│   ├── agent.py                     # EvaluatorAgent orchestrator
│   └── models.py                    # FindingEvaluation, ClauseEvaluation, EvaluationResult
│
├── output_agent/                    # STAGE 5 — Report generation
│   ├── __init__.py
│   ├── agent.py                     # OutputAgent orchestrator
│   └── models.py                    # ReportSection, FinalReport
│
├── sample_docs/
│   └── sample_nda.pdf               # Generated NDA with known risky clauses
│
├── legal_corpus/
│   └── vectordb/                    # ChromaDB persistent storage (auto-created)
│
└── output/                          # Pipeline output directory (auto-created)
    ├── intake_result.json
    ├── research_result.json
    ├── analysis_result.json
    ├── evaluation_result.json
    ├── final_report.json
    └── final_report.md
```

**Total: 25 Python files, ~4,628 lines of code, 23 Pydantic models**

---

## Stage-by-stage implementation details

### Stage 1: Intake Agent

**Purpose:** Take a raw PDF and produce structured, classified clauses.

**Pipeline:** PDF → pdfplumber extraction → regex-based heading detection → clause splitting → LLM classification → metadata extraction

**Key components:**

- `PDFExtractor` — Uses pdfplumber (chosen over PyPDF2 for better reading-order preservation). Cleans boilerplate footers, strips standalone page numbers, normalizes whitespace. Tracks page boundaries for clause attribution.

- `ClauseSegmenter` — Rule-based first pass using 7 regex patterns that match common legal heading formats: numbered sections ("1. Definitions"), article format ("ARTICLE II"), decimal sub-sections ("3.2.1"), lettered sections ("A. Representations"), ALL-CAPS headings, and exhibit/schedule markers. Also handles preamble detection before the first heading and merges very short clauses (< 50 chars) into predecessors. This gets ~80% of contracts right without spending LLM tokens.

- `ClauseClassifier` — Sends clause batches to Claude API with a structured system prompt. Returns clause_type (one of 25 categories), risk_level (high/medium/low/none), plain-language summary, key_terms, and flags. Batches up to 15 clauses per API call to save cost. Separate API call for document metadata (type, parties, date, jurisdiction) using only the preamble.

**25 clause types supported:** definitions, confidentiality, non_disclosure, obligations, indemnification, termination, governing_law, dispute_resolution, liability, intellectual_property, representations, non_compete, non_solicitation, payment_terms, scope_of_work, force_majeure, assignment, severability, entire_agreement, amendments, notices, preamble, signature_block, miscellaneous, unknown

**Risk classification criteria:**
- HIGH: One-sided indemnification, unlimited liability, overly broad non-competes, automatic renewal with no exit, unilateral amendment rights, broad IP assignment
- MEDIUM: Standard but negotiable terms, moderately broad restrictions, short cure periods, ambiguous language
- LOW: Balanced/mutual obligations, standard protective language, clear and fair terms
- NONE: Preamble, definitions, signature blocks, procedural clauses

**Verified working:** Tested on sample NDA — correctly extracted 3 pages, segmented into 10 clauses with accurate heading detection and page-span attribution.

---

### Stage 2: Research Agent

**Purpose:** Ground each clause in actual legal authority via RAG.

**Pipeline:** Clause → LLM query generation (3-5 queries per clause) → multi-query ChromaDB retrieval → jurisdiction-aware filtering → LLM synthesis

**Key components:**

- `LegalDocumentLoader` — Chunks legal texts with configurable chunk_size (800 chars default) and overlap (150 chars). Supports three ingestion formats: structured JSON (with section-level metadata), plain text files, and inline data. Splits on paragraph boundaries, not mid-sentence.

- `LegalVectorStore` — ChromaDB wrapper with persistent storage. Ships with `LocalHashEmbedding` — a 384-dimension hash-based embedding function using word unigrams/bigrams with TF normalization. This is a FALLBACK for offline environments. **For production, swap to sentence-transformers or Voyage embeddings — one-line change in the constructor.**

- `ResearchSynthesizer` — Two LLM calls per clause: (1) query generation that produces diverse search angles from clause text + type + jurisdiction, (2) synthesis that takes the retrieved sources and produces legal_context, applicable_law, risk_analysis, recommendations, and enforceability_notes.

- `corpus_seeder.py` — Seeds vector store with 7 curated legal sources (15 chunks total) covering: Delaware restrictive covenant standards (non-compete enforceability, blue pencil doctrine, non-solicitation distinction), indemnification standards, limitation of liability, Delaware Uniform Trade Secrets Act, IP assignment in NDAs, termination provisions, Delaware choice of law.

**Key design decision — multi-query retrieval:** Rather than searching with raw clause text (matches on writing style, not legal concepts), the agent generates 3-5 targeted queries per clause. For a non-compete clause, this produces queries like "non-compete enforceability Delaware", "restrictive covenant reasonable scope", "blue pencil doctrine". Each query retrieves top 3 results independently, results are deduplicated by ID. If jurisdiction-filtered search returns < 3 sources, automatically broadens by dropping the filter.

**Embedding swap instructions:** In `vector_store.py`, replace `LocalHashEmbedding(dim=384)` with:
```python
# Option 1: sentence-transformers (local, free)
from chromadb.utils import embedding_functions
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Option 2: OpenAI
ef = embedding_functions.OpenAIEmbeddingFunction(api_key="...", model_name="text-embedding-3-small")
```

**Verified working:** Corpus seeds correctly (15 chunks, 2 jurisdictions). Search returns relevant results — "non-compete enforceability duration" retrieves the Delaware restrictive covenant standards; "indemnification one-sided" retrieves the indemnification treatise; "IP assignment NDA" retrieves the IP assignment source.

---

### Stage 3: Analysis Agent

**Purpose:** Merge intake classification with research findings to produce actionable legal analysis.

**Pipeline:** (IntakeResult + ResearchResult) → match clause to research → per-clause deep analysis → missing clause detection → cross-clause check → document scoring

**Key components:**

- `ClauseAnalyzer` — Two LLM calls: (1) per-clause analysis that receives clause text, intake flags, and full research synthesis — produces structured Findings with severity, category, legal_basis, and suggested_revision; (2) document-level analysis that receives all clause summaries — produces missing clauses, cross-clause issues, and overall scoring.

**Finding severity levels:**
- CRITICAL: Likely unenforceable or severely prejudicial
- MAJOR: Significant risk, strongly recommend change
- MODERATE: Notable concern, negotiation recommended
- MINOR: Suboptimal but low practical impact
- INFO: Informational observation

**9 issue categories:** overbroad, one_sided, unenforceable, ambiguous, missing_protection, non_standard, unconscionable, conflict, compliance

**Document scoring:** 0-100 scale weighted toward critical and major findings. Produces a concrete proceed_recommendation: "sign as-is", "sign with minor revisions", "negotiate significant changes", "do not sign without major revisions", or "walk away".

**The three-way merge is the core value:** The Analysis Agent doesn't re-read the document or re-search. It operates on structured outputs from both upstream agents, combining clause text + intake flags + research synthesis + applicable law into findings that cite actual retrieved sources with concrete revision language.

---

### Stage 4: Evaluator Agent

**Purpose:** Quality gate — catch errors before the user sees them.

**Two-pass evaluation:**

1. **Per-clause evaluation:** Each clause's findings are checked against the research that was available. For each finding: is the legal_basis actually supported by retrieved sources? Is severity proportionate? Is the suggested revision actionable? Findings score high/medium/low/failed.

2. **Document-level validation:** Is the risk score consistent with findings? Are there hallucinated citations? Do cross-clause issues hold up?

**Reflection loop trigger:** If any finding scores "failed" or 2+ score "low", the clause gets flagged for reanalysis. The pipeline currently logs the flag and proceeds — the re-analysis execution path is wired in the data model but the retry logic in main.py is not yet implemented.

**Key metric:** `pass_rate` — percentage of clauses achieving high or medium confidence.

---

### Stage 5: Output Agent

**Purpose:** Transform raw analysis into a client-ready deliverable.

**Report sections:**
1. Executive summary (LLM-generated, plain language, targets non-lawyers)
2. Risk scorecard (score, findings breakdown, recommendation)
3. Clause-by-clause findings (sorted by severity, with legal basis and suggested revisions)
4. Missing protective clauses (with draft language)
5. Cross-clause issues
6. Negotiation playbook (prioritized critical/major items with concrete revision language)
7. Confidence disclosure (evaluator results + "this is not legal advice" disclaimer)

**Dual output:** Structured JSON (FinalReport) for programmatic consumption + compiled Markdown for human reading / PDF conversion.

---

## Sample NDA test document

The `generate_sample_nda.py` script creates a 3-page NDA between Acme Technologies Inc. (Delaware) and Beta Solutions LLC (California) with these intentionally risky provisions:

1. **Non-compete (Section 3):** 5-year duration, worldwide geographic scope, covers all competitive activity — all overbroad per Delaware standards
2. **IP assignment (Section 4):** Broad assignment of any "ideas, concepts, inventions" during the agreement term — atypical and overreaching for an NDA
3. **Indemnification (Section 5):** One-sided, Receiving Party only, survives "without limitation" — disfavored by courts
4. **Liability cap (Section 6):** $100 cap for Disclosing Party, no cap for Receiving Party — likely unconscionable
5. **Termination (Section 7):** Only Disclosing Party can terminate without cause, immediate termination without cure period for breach, 7-year confidentiality survival

The pipeline should flag all five as critical or high-risk findings with specific legal bases from the corpus.

---

## Current capabilities assessment

### Fully working
- Complete 5-stage pipeline from PDF to Markdown report
- CLI entry point with subcommands (intake, pipeline, seed-corpus)
- 25 clause type classifications
- ChromaDB vector store with persistent storage
- LLM-powered query generation, classification, analysis, evaluation, and report writing
- Structured JSON output at every stage
- Rich console output with formatted tables
- API retry logic with rate limit handling
- Sample NDA generator for testing
- Per-stage JSON persistence in output/ directory

### Scaffolded (data flow wired, execution not implemented)
- **Reflection loop execution:** Evaluator correctly flags clauses, but main.py logs the flag rather than re-running the Analysis Agent. The ClauseAnalyzer accepts feedback context — needs the retry loop in the pipeline.
- **DAG parallelization:** Pipeline runs sequentially. Agent interfaces support per-clause execution. Parallelizing via asyncio or task queue is straightforward.

### Not built yet (needed for production/demo expansion)
- **Real legal corpus:** 15 curated chunks vs. thousands needed. Ingestion pipeline is built, needs real statute/case law data.
- **Production embeddings:** Hash-based fallback. Swap one line to sentence-transformers or Voyage.
- **Web UI/dashboard:** CLI only. Needs Streamlit or React frontend.
- **OCR/scanned PDF support:** Text-based PDFs only. Needs pytesseract preprocessing.
- **Multi-document types:** NDA only. See scope expansion section below.
- **Evaluation benchmarks:** No automated test suite. Needs annotated contracts with known issues.

---

## Scope: what's covered vs. what's in the presentation

### Currently implemented: NDAs only

The entire pipeline — clause types, corpus, segmentation, prompts, sample document — is tuned for NDAs and closely related confidentiality agreements.

### Presentation scope (not yet built)

The presentation describes six legal specializations:

| Specialization | What it needs | Effort |
|---|---|---|
| **General contracts** (service, consulting, SaaS) | New clause types (SLAs, warranties, acceptance criteria, auto-renewal), UCC corpus | Medium — closest extension |
| **Employment agreements** | Employment-specific clauses (at-will, equity vesting, benefits), state-specific employment law corpus (CA vs TX vs NY non-compete rules) | Medium |
| **IP/Patents** | Fundamentally different pipeline — claims interpretation, prior art, prosecution history. Current system handles IP assignment *clauses within contracts* but not patent review. | High — different architecture |
| **Compliance** | Regulatory corpus (GDPR, HIPAA, SOX), analysis frame that maps obligations to regulations rather than assessing enforceability | Medium-High |
| **Legal advice** | Deepest corpus, jurisdiction-specific analysis, human-in-the-loop validation. System currently disclaims "this is not legal advice" | High — sensitivity concerns |
| **Litigation support** | Document organization at scale, issue spotting across document sets, timeline construction | High — multi-document architecture |

**Recommended next expansion:** General contracts — adds a second document type with the most code reuse.

---

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run full pipeline on sample NDA
python main.py pipeline --sample

# Run full pipeline on your own document
python main.py pipeline path/to/contract.pdf

# Only research high-risk clauses
python main.py pipeline --sample --risk-threshold high

# Run intake only
python main.py intake --sample

# Seed the legal corpus
python main.py seed-corpus
```

**Output files produced by `pipeline` command:**
- `output/intake_result.json` — Clause classifications
- `output/research_result.json` — Retrieved sources and synthesis
- `output/analysis_result.json` — Findings, scoring, missing clauses
- `output/evaluation_result.json` — Confidence scores, hallucination flags
- `output/final_report.json` — Structured report
- `output/final_report.md` — Human-readable Markdown report

---

## Dependencies

```
anthropic>=0.39.0       # Claude API client
pdfplumber>=0.11.0      # PDF text extraction
pydantic>=2.0.0         # Data models with validation
rich>=13.0.0            # Formatted console output
chromadb>=0.5.0         # Vector store for RAG
reportlab               # Sample NDA PDF generation (dev only)
```

**All LLM calls use `claude-sonnet-4-20250514` by default.** Configurable via `--model` flag.

---

## Key design decisions and rationale

1. **Rule-based segmentation before LLM classification:** Saves tokens and makes classification more reliable. The regex segmenter handles ~80% of contracts. The LLM refines, not replaces.

2. **Separate query generation from retrieval:** Asking Claude to generate search queries (instead of searching with raw clause text) dramatically improves recall by approaching the same legal concept from multiple angles.

3. **Hash-based embedding fallback:** The default ChromaDB embedding (ONNX MiniLM) requires downloading a model from HuggingFace, which fails in network-restricted environments. The LocalHashEmbedding produces 384-dim vectors via feature hashing — good enough for demo, one-line swap for production.

4. **Pydantic models at every boundary:** Every agent-to-agent handoff uses typed Pydantic models with JSON serialization. This makes the pipeline debuggable (inspect any intermediate JSON), testable (mock any stage), and extensible (add fields without breaking downstream).

5. **Evaluator as a separate agent (not inline checks):** Having a dedicated evaluation stage with its own LLM call catches issues that inline checks miss — like the Analysis Agent fabricating a legal citation that sounds plausible. The evaluator cross-references findings against actual retrieved sources.

6. **Sequential pipeline (not parallel yet):** Simpler to debug, test, and demo. The per-clause interfaces are designed for parallelization — each clause can be researched and analyzed independently. The DAG conversion is an optimization step, not a correctness requirement.

---

## Implementation priorities for Claude Code

### Priority 1: Make it demo-ready
1. Implement the reflection loop execution in main.py (re-run Analysis Agent on flagged clauses)
2. Swap to real embeddings (sentence-transformers or Voyage)
3. Expand the legal corpus with 50-100 real statute sections
4. Add a Streamlit UI for PDF upload and report viewing

### Priority 2: Expand document type coverage
1. Add general contract support (service agreements, consulting agreements)
2. Expand clause types for contract-specific provisions
3. Add corpus coverage for UCC and general commercial contract law
4. Test against 5-10 real contracts of varying quality

### Priority 3: Production hardening
1. Async pipeline with per-clause parallelization
2. Evaluation benchmark suite with annotated test contracts
3. Cost tracking per pipeline run (token usage)
4. Error recovery (resume from last successful stage)
5. OCR preprocessing for scanned PDFs

### Priority 4: Presentation alignment
1. Employment agreement support
2. Compliance checking capability
3. Multi-document analysis (compare two versions of a contract)
4. Web dashboard with real-time pipeline progress

---

## Technical notes for Claude Code

- **Python 3.12+** required (uses `from __future__ import annotations` for modern type hints)
- **All agents use the same API pattern:** constructor takes `api_key`, `model`, `verbose`; main method takes upstream result(s) and returns a typed result
- **The `sys.path.insert` in agent.py files** is for cross-package imports — may need adjustment depending on your working directory in Claude Code
- **ChromaDB creates the vectordb directory on first run** — if you get collection conflicts, delete `legal_corpus/vectordb/` and re-seed
- **Rate limiting:** All API calls have retry logic with exponential backoff. Default 3 retries. At ~8-12 LLM calls per pipeline run, a single NDA review costs approximately $0.10-0.30 in API credits with Sonnet.
