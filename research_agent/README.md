# Research Agent (Stage 2)

The Research Agent grounds each clause in actual legal authority via RAG (Retrieval-Augmented Generation). It takes an `IntakeResult` and produces a `ResearchResult` with relevant legal sources, applicable law, risk analysis, and recommendations for each clause.

## What It Does

```
IntakeResult (classified clauses)
    |
    v
For each clause (prioritized by risk):
    +-------------------+     +-------------------+     +-------------------+
    | Query Generation  | --> | RAG Retrieval     | --> | Legal Synthesis   |
    | LLM: 3-5 queries  |     | ChromaDB search   |     | LLM: synthesize  |
    | per clause         |     | multi-query +     |     | sources into      |
    |                    |     | jurisdiction filter|     | actionable research|
    +-------------------+     +-------------------+     +-------------------+
    |
    v
ResearchResult (per-clause sources + legal context + recommendations)
```

## Components

### ResearchAgent (`agent.py`)

The orchestrator that coordinates the full research pipeline:

1. Initializes the ChromaDB vector store (auto-seeds if empty)
2. Filters and prioritizes clauses by risk level (skips preamble/signature blocks)
3. For each clause: generates queries, retrieves sources, synthesizes research
4. Packages results into a `ResearchResult`

Supports a `risk_threshold` parameter:
- `"high"` — only research high-risk clauses
- `"medium"` — research high + medium risk
- `"low"` (default) — research all substantive clauses

### ResearchSynthesizer (`research_synthesizer.py`)

The "brain" of the Research Agent. Makes two LLM calls per clause:

**Call 1 — Query generation:**
Given clause text, type, jurisdiction, and risk level, generates 3-5 diverse search queries. For example, a non-compete clause produces queries like:
- "non-compete enforceability Delaware"
- "restrictive covenant reasonable scope duration"
- "blue pencil doctrine"
- "non-compete limitations NDA context"

This multi-angle approach dramatically improves recall over searching with raw clause text.

**Call 2 — Source synthesis:**
Given the clause and retrieved sources, produces:
- `legal_context` — 2-3 paragraph legal framework overview
- `applicable_law` — specific statutes/regulations that apply
- `risk_analysis` — how the clause measures against legal standards
- `recommendations` — concrete, negotiation-ready suggestions
- `enforceability_notes` — likely enforceability given jurisdiction and terms

### LegalVectorStore (`vector_store.py`)

ChromaDB-backed persistent vector store for legal document search.

**Key features:**
- **Persistent storage** — corpus survives restarts (`legal_corpus/vectordb/`)
- **Multi-query retrieval** — searches with each generated query independently, deduplicates results by source ID, sorts by best relevance score
- **Jurisdiction-aware filtering** — filters results by jurisdiction; automatically broadens the search if too few results are found
- **Metadata filtering** — can filter by source type (statute, case law, regulation, etc.)

**LocalHashEmbedding (fallback):**
Ships with a 384-dimension hash-based embedding function using word unigrams/bigrams with TF normalization and L2 normalization. This is a **fallback for offline environments**. For production, swap one line:

```python
# In vector_store.py, replace LocalHashEmbedding with:

# Option 1: sentence-transformers (local, free)
from chromadb.utils import embedding_functions
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Option 2: OpenAI embeddings
ef = embedding_functions.OpenAIEmbeddingFunction(api_key="...", model_name="text-embedding-3-small")
```

### LegalDocumentLoader (`document_loader.py`)

Loads, chunks, and prepares legal documents for vector store ingestion.

**Chunking strategy:**
- Target chunk size: 800 characters (configurable)
- Overlap: 150 characters between consecutive chunks
- Splits on paragraph boundaries, not mid-sentence
- Attaches metadata to each chunk (source type, jurisdiction, title, section ID, heading)

**Supported input formats:**
- **Structured JSON** — with section-level metadata (source type, jurisdiction, sections)
- **Plain text files** — auto-detects section headings
- **Inline data** — for programmatic seeding (used by the corpus seeder)

### Corpus Seeder (`corpus_seeder.py`)

Seeds the vector store with 7 curated legal sources (15 chunks total) covering:

| Source | Jurisdiction | Topics |
|--------|-------------|--------|
| Delaware Restrictive Covenant Standards | Delaware | Non-compete enforceability, blue pencil doctrine, non-solicitation distinction |
| Indemnification Clause Standards | General | Mutual vs one-sided indemnification, survival periods |
| Limitation of Liability | General | Enforceability of liability caps, consequential damages waivers |
| Delaware Uniform Trade Secrets Act | Delaware | Trade secret definition, remedies for misappropriation |
| IP Assignment in NDAs | General | Scope of IP assignment, work made for hire vs assignment |
| Termination Provisions | General | Unilateral termination rights, survival clauses |
| Delaware Choice of Law | Delaware | Choice of law enforceability, fee-shifting provisions |

In production, this would be replaced with actual statute databases, case law APIs, or curated PDFs.

## Data Models (`models.py`)

| Model | Description |
|-------|-------------|
| `SourceType` | Enum: `statute`, `case_law`, `regulation`, `legal_treatise`, `restatement`, `model_code`, `commentary`, `unknown` |
| `LegalSource` | A single retrieved reference: source ID, type, title, jurisdiction, text, relevance score, metadata |
| `ClauseResearch` | Research for one clause: sources, legal context, applicable law, risk analysis, recommendations, enforceability notes, queries used |
| `ResearchResult` | Full output: document info, list of clause research, corpus stats |

## Usage

```python
from intake_agent import IntakeAgent
from research_agent import ResearchAgent

intake = IntakeAgent(api_key="sk-ant-...")
intake_result = intake.process("contract.pdf")

research = ResearchAgent(api_key="sk-ant-...")
research_result = research.research(intake_result)
research_result.save("output/research_result.json")

# Access high-risk research
for cr in research_result.high_risk_research():
    print(f"{cr.clause_heading}: {len(cr.sources)} sources, {len(cr.recommendations)} recs")
```

## Output

The `ResearchResult` is passed to the Analysis Agent (Stage 3). It provides the legal grounding that allows the Analysis Agent to produce findings that cite actual retrieved sources rather than hallucinating legal citations.

## Files

```
research_agent/
├── __init__.py
├── agent.py                # ResearchAgent orchestrator
├── models.py               # Pydantic data models
├── document_loader.py      # Chunks legal texts for vector store ingestion
├── vector_store.py         # ChromaDB wrapper + LocalHashEmbedding fallback
├── research_synthesizer.py # LLM query generation + source synthesis
└── corpus_seeder.py        # Sample legal corpus (7 sources, 15 chunks)
```
