# Agentic Lawyer AI

A multi-agent AI system that autonomously reviews legal documents, retrieves applicable law, flags risks, and generates plain-language reports. Built as a 5-agent pipeline orchestrated in a DAG structure with RAG grounding and a reflection loop for quality assurance.

**Team:** Kevin Kostage, Ping Liu, Adib Bazgir, Mohsen Rezaei, Saiteja Labba

**Core thesis:** Legal document review encompasses diverse specialized tasks more efficiently performed by dedicated agents than a single LLM. Breaking the work into intake, research, analysis, evaluation, and reporting allows each agent to focus on what it does best while passing structured data downstream.

---

## Pipeline Overview

```
PDF Document
    |
    v
+-------------------------------------------------------------+
| STAGE 1: INTAKE AGENT                                       |
|   PDF extraction -> clause segmentation -> LLM classif.     |
|   Output: IntakeResult (metadata + typed clauses + flags)    |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| STAGE 2: RESEARCH AGENT                                     |
|   Query generation -> RAG retrieval -> legal synthesis       |
|   Output: ResearchResult (sources + applicable law + recs)   |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| STAGE 3: ANALYSIS AGENT                                     |
|   Merge intake + research -> deep findings -> document score |
|   Output: AnalysisResult (findings + missing clauses + score)|
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| STAGE 4: EVALUATOR AGENT                                    |
|   Grounding check -> confidence scoring -> reflection loop   |
|   Output: EvaluationResult (confidence + hallucination flags)|
|   Re-analysis trigger if low confidence                      |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| STAGE 5: OUTPUT AGENT                                       |
|   Executive summary -> report sections -> negotiation tips   |
|   Output: FinalReport (JSON + Markdown)                      |
+-------------------------------------------------------------+
```

---

## Scope

The pipeline is currently tuned for **NDAs and confidentiality agreements**. It supports 25 clause types (definitions, confidentiality, non-disclosure, obligations, indemnification, termination, governing law, dispute resolution, liability, intellectual property, representations, non-compete, non-solicitation, payment terms, scope of work, force majeure, assignment, severability, entire agreement, amendments, notices, preamble, signature block, miscellaneous, unknown) and classifies each clause by risk level (high, medium, low, none).

A sample NDA (`generate_sample_nda.py`) is included with intentionally risky provisions (5-year worldwide non-compete, one-sided indemnification, unconscionable liability cap) to validate that the pipeline catches critical issues.

---

## The 5 Agents

### Stage 1: Intake Agent

**Purpose:** Parse a raw PDF into structured, classified clauses that downstream agents can consume.

**How it works:**
1. **PDF Extraction** (`pdf_extractor.py`) — Uses pdfplumber to extract text with page-number tracking. Removes boilerplate headers/footers and normalizes whitespace.
2. **Clause Segmentation** (`clause_segmenter.py`) — Rule-based heading detection using 7 regex patterns (numbered sections, article format, decimal sub-sections, lettered sections, ALL-CAPS headings, exhibit markers). Handles preamble detection and short-clause merging. Gets ~80% of contracts right without LLM tokens.
3. **LLM Classification** (`clause_classifier.py`) — Sends clause batches to Claude API. Each clause receives a type, risk level, plain-language summary, key terms, and flags. Batches up to 15 clauses per call.
4. **Metadata Extraction** — Separate LLM call on the preamble to extract document type, parties, effective date, and jurisdiction.

**Output:** `IntakeResult` — metadata, list of typed/classified clauses, raw text.

---

### Stage 2: Research Agent

**Purpose:** Ground each clause in actual legal authority via RAG retrieval.

**How it works:**
1. **Query Generation** — LLM generates 3-5 targeted search queries per clause (e.g., for a non-compete clause: "non-compete enforceability Delaware", "restrictive covenant reasonable scope", "blue pencil doctrine"). This approach yields far better recall than searching with raw clause text.
2. **RAG Retrieval** — Each query hits a ChromaDB vector store seeded with curated legal sources. Top results are retrieved per query and deduplicated. Jurisdiction-aware filtering automatically broadens if too few results are found.
3. **Legal Synthesis** — LLM combines retrieved sources into legal context, applicable law, risk analysis, recommendations, and enforceability notes.

The vector store ships with a `LocalHashEmbedding` fallback (384-dim hash-based vectors) for offline environments. For production, this is a one-line swap to sentence-transformers or Voyage embeddings.

**Output:** `ResearchResult` — per-clause legal sources, applicable law, risk analysis, and recommendations.

---

### Stage 3: Analysis Agent

**Purpose:** Merge intake classification with research findings to produce actionable legal analysis.

**How it works:**
1. **Per-Clause Deep Analysis** — Receives clause text, intake flags, and research synthesis. Produces structured findings with severity (critical/major/moderate/minor/info), issue category (overbroad, one-sided, unenforceable, ambiguous, etc.), legal basis, and suggested revision language.
2. **Missing Clause Detection** — Identifies protective clauses that should be present but aren't (e.g., a missing dispute resolution clause).
3. **Document-Level Scoring** — Produces a 0-100 risk score weighted toward critical/major findings and a concrete recommendation: "sign as-is", "negotiate changes", "do not sign", or "walk away".

The three-way merge is the core value of this agent: it doesn't re-read the document or re-search. It operates on structured outputs from both upstream agents, combining clause text + intake flags + research synthesis into findings that cite retrieved sources.

**Output:** `AnalysisResult` — per-clause findings, missing clauses, document score, and proceed recommendation.

---

### Stage 4: Evaluator Agent

**Purpose:** Quality gate that validates analysis output before it reaches the user.

**How it works:**
1. **Per-Clause Evaluation** — Checks each finding against the research that was available. Is the legal basis supported by retrieved sources? Is severity proportionate? Is the suggested revision actionable? Each finding scores high/medium/low/failed.
2. **Document-Level Validation** — Checks whether the risk score is consistent with findings and flags hallucinated citations.
3. **Reflection Loop** — If any finding scores "failed" or 2+ score "low", the clause is flagged for re-analysis. This catches LLM hallucinations that inline checks would miss.

**Key metric:** `pass_rate` — percentage of clauses achieving high or medium confidence.

**Output:** `EvaluationResult` — confidence scores, hallucination flags, and list of clauses needing re-analysis.

---

### Stage 5: Output Agent

**Purpose:** Transform raw analysis into a client-ready deliverable.

**Report sections generated:**
1. Executive summary (plain language, targeting non-lawyers)
2. Risk scorecard (score, findings breakdown, recommendation)
3. Clause-by-clause findings (sorted by severity, with legal basis and suggested revisions)
4. Missing protective clauses (with draft language)
5. Cross-clause issues
6. Negotiation playbook (prioritized items with concrete revision language)
7. Confidence disclosure (evaluator results + "this is not legal advice" disclaimer)

**Output:** `FinalReport` — structured JSON for programmatic use + compiled Markdown for human reading.

---

## General Flow

1. User provides a PDF legal document (or uses the included sample NDA)
2. The **Intake Agent** extracts text, segments it into clauses, classifies each by type and risk, and extracts document metadata
3. The **Research Agent** takes the classified clauses and retrieves relevant legal authorities from a ChromaDB vector store, then synthesizes findings per clause
4. The **Analysis Agent** merges the intake classifications with research findings to produce detailed risk findings, identifies missing clauses, and scores the document overall
5. The **Evaluator Agent** validates the analysis output — checking that findings are grounded in retrieved sources, flagging hallucinations, and triggering re-analysis where confidence is low
6. The **Output Agent** compiles everything into a structured report with an executive summary, risk scorecard, clause-by-clause findings, and negotiation playbook

All agents share a common pattern: constructor takes `api_key`, `model`, `verbose`; main method takes upstream result(s) and returns a typed Pydantic model. Every agent-to-agent handoff uses typed models with JSON serialization, making the pipeline debuggable (inspect any intermediate JSON), testable (mock any stage), and extensible.

---

## Project Structure

```
agentic_lawyer_complete/
├── main.py                          # CLI entry point — orchestrates all 5 stages
├── requirements.txt                 # anthropic, pdfplumber, pydantic, rich, chromadb
├── generate_sample_nda.py           # Creates test PDF with intentionally risky clauses
│
├── intake_agent/                    # STAGE 1 — Parse, segment, classify
│   ├── agent.py                     # IntakeAgent orchestrator
│   ├── models.py                    # RawPage, Clause, ClauseType, RiskLevel, IntakeResult
│   ├── pdf_extractor.py             # pdfplumber text extraction + cleaning
│   ├── clause_segmenter.py          # Rule-based heading detection (7 regex patterns)
│   └── clause_classifier.py         # LLM classification via Claude API
│
├── research_agent/                  # STAGE 2 — RAG retrieval + synthesis
│   ├── agent.py                     # ResearchAgent orchestrator
│   ├── models.py                    # LegalSource, ClauseResearch, ResearchResult
│   ├── document_loader.py           # Chunks legal texts for vector store ingestion
│   ├── vector_store.py              # ChromaDB wrapper + LocalHashEmbedding fallback
│   ├── research_synthesizer.py      # LLM query generation + source synthesis
│   └── corpus_seeder.py             # Sample legal corpus (7 sources, 15 chunks)
│
├── analysis_agent/                  # STAGE 3 — Deep analysis + scoring
│   ├── agent.py                     # AnalysisAgent orchestrator
│   ├── models.py                    # Finding, ClauseAnalysis, MissingClause, DocumentScore
│   └── clause_analyzer.py           # LLM clause analysis + document-level analysis
│
├── evaluator_agent/                 # STAGE 4 — Quality gate + reflection
│   ├── agent.py                     # EvaluatorAgent orchestrator
│   └── models.py                    # FindingEvaluation, ClauseEvaluation, EvaluationResult
│
├── output_agent/                    # STAGE 5 — Report generation
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

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Run the full pipeline on the sample NDA
python main.py pipeline --sample

# 4. Or process your own document
python main.py pipeline path/to/contract.pdf

# 5. Only research high-risk clauses
python main.py pipeline --sample --risk-threshold high

# 6. Run intake stage only
python main.py intake --sample

# 7. Seed the legal corpus separately
python main.py seed-corpus
```

**Output files produced by the `pipeline` command:**

| File | Contents |
|------|----------|
| `output/intake_result.json` | Clause classifications and metadata |
| `output/research_result.json` | Retrieved legal sources and synthesis |
| `output/analysis_result.json` | Findings, scoring, missing clauses |
| `output/evaluation_result.json` | Confidence scores, hallucination flags |
| `output/final_report.json` | Structured report (programmatic use) |
| `output/final_report.md` | Human-readable Markdown report |

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

All LLM calls use `claude-sonnet-4-20250514` by default. Configurable via the `--model` flag.

---

## Key Design Decisions

1. **Rule-based segmentation before LLM** — Regex patterns handle ~80% of contracts without spending tokens. The LLM refines, not replaces.
2. **Multi-query retrieval** — Generating 3-5 search angles per clause dramatically improves recall over searching with raw clause text.
3. **Hash-based embedding fallback** — `LocalHashEmbedding` works offline. One-line swap to sentence-transformers or Voyage for production.
4. **Pydantic models at every boundary** — Typed, serializable, debuggable, extensible. Inspect any intermediate JSON to debug the pipeline.
5. **Evaluator as a separate agent** — A dedicated evaluation stage catches hallucinations that inline checks miss by cross-referencing findings against retrieved sources.
6. **Sequential pipeline** — Simpler to debug and demo. Per-clause interfaces are designed for future parallelization via asyncio.
