# Analysis Agent (Stage 3)

The Analysis Agent merges intake classifications with research findings to produce actionable legal analysis. It takes both the `IntakeResult` and `ResearchResult` and produces an `AnalysisResult` with per-clause findings, missing clause detection, cross-clause issue identification, and overall document scoring.

## What It Does

```
IntakeResult + ResearchResult
    |
    v
+-----------------------------------------+
| For each clause:                        |
|   clause text + intake flags            |
|   + research synthesis + applicable law |
|   --> LLM deep analysis                 |
|   --> Findings with severity, legal     |
|       basis, suggested revisions        |
+-----------------------------------------+
    |
    v
+-----------------------------------------+
| Document-level analysis:                |
|   All clause summaries --> LLM          |
|   --> Missing clauses                   |
|   --> Cross-clause issues               |
|   --> Risk score (0-100)                |
|   --> Proceed recommendation            |
+-----------------------------------------+
    |
    v
AnalysisResult
```

## How the Three-Way Merge Works

This is the core value of the multi-agent architecture. The Analysis Agent does **not** re-read the document or re-search legal sources. Instead, it operates on the structured outputs from both upstream agents:

1. **From Intake Agent** — clause text, heading, type, risk level, flags, summary
2. **From Research Agent** — legal context, risk analysis, recommendations, enforceability notes, applicable law

The LLM receives all of this combined context for each clause and produces findings that cite the actual retrieved sources with concrete revision language. This three-way merge (clause text + intake classification + research synthesis) is what allows the system to produce grounded analysis rather than generic observations.

## Components

### AnalysisAgent (`agent.py`)

The orchestrator that runs a two-step pipeline:

**Step 1 — Per-clause analysis:**
- Matches each clause to its research by index
- Skips preamble and signature blocks
- Calls the `ClauseAnalyzer` with combined intake + research data
- Reports inline feedback on severity counts

**Step 2 — Document-level analysis:**
- Sends all clause summaries to the LLM
- Identifies missing protective clauses
- Detects cross-clause contradictions
- Produces overall risk scoring and proceed recommendation

### ClauseAnalyzer (`clause_analyzer.py`)

Makes two LLM calls:

**Call 1 — Per-clause deep analysis:**
Receives clause text, intake flags, and full research synthesis. Produces:
- `revised_risk_level` — may upgrade/downgrade the intake classification
- `findings[]` — structured issues, each with severity, category, legal basis, and suggested revision
- `overall_assessment` — 2-3 paragraph grounded analysis
- `negotiation_strategy` — practical advice for negotiating improvements
- `market_comparison` — how the clause compares to standard market terms

**Call 2 — Document-level analysis:**
Receives summaries of all clause analyses. Produces:
- `missing_clauses[]` — protective clauses the document should have
- `document_score` — overall risk scoring (0-100)
- `cross_clause_issues[]` — issues spanning multiple clauses

## Finding Severity Levels

| Severity | Meaning |
|----------|---------|
| **CRITICAL** | Clause is likely unenforceable, unconscionable, or creates severe legal exposure |
| **MAJOR** | Significant risk that strongly warrants renegotiation |
| **MODERATE** | Notable concern worth raising in negotiation |
| **MINOR** | Suboptimal but low practical impact |
| **INFO** | Informational observation, no action needed |

## Issue Categories

| Category | Description |
|----------|-------------|
| `overbroad` | Scope too wide (geographic, temporal, activity) |
| `one_sided` | Favors one party disproportionately |
| `unenforceable` | Likely invalid under applicable law |
| `ambiguous` | Unclear language creates uncertainty |
| `missing_protection` | Expected safeguard absent |
| `non_standard` | Deviates from market practice |
| `unconscionable` | May be struck by a court |
| `conflict` | Internal contradiction with other clauses |
| `compliance` | Potential regulatory issue |

## Document Scoring

- **Scale:** 0-100 (higher = more risky)
- **Weighting:** Critical and major findings are weighted heavily. Multiple critical findings push the score above 70.
- **Recommendations:** `sign as-is`, `sign with minor revisions`, `negotiate significant changes`, `do not sign without major revisions`, `walk away`

## Data Models (`models.py`)

| Model | Description |
|-------|-------------|
| `Severity` | Enum: `critical`, `major`, `moderate`, `minor`, `info` |
| `IssueCategory` | Enum: 9 categories (overbroad, one_sided, unenforceable, etc.) |
| `Finding` | Severity, category, title, description, legal basis, suggested revision |
| `ClauseAnalysis` | Per-clause: findings, revised risk, overall assessment, negotiation strategy, market comparison |
| `MissingClause` | Type, importance, description, suggested language |
| `DocumentScore` | Overall risk, score (0-100), summary, top concerns, strengths, proceed recommendation |
| `AnalysisResult` | Full output: clause analyses, missing clauses, document score, cross-clause issues |

## Usage

```python
from analysis_agent import AnalysisAgent

agent = AnalysisAgent(api_key="sk-ant-...")
analysis_result = agent.analyze(intake_result, research_result)
analysis_result.save("output/analysis_result.json")

# Access critical findings
for heading, finding in analysis_result.critical_findings():
    print(f"{heading}: {finding.title}")
    print(f"  Legal basis: {finding.legal_basis}")
    print(f"  Suggested: {finding.suggested_revision}")

# Check document score
ds = analysis_result.document_score
print(f"Risk: {ds.overall_risk} ({ds.score}/100) - {ds.proceed_recommendation}")
```

## Output

The `AnalysisResult` is passed to both the Evaluator Agent (Stage 4) for quality validation and the Output Agent (Stage 5) for report generation.

## Files

```
analysis_agent/
├── __init__.py
├── agent.py            # AnalysisAgent orchestrator (analyze method)
├── models.py           # Pydantic data models (Finding, ClauseAnalysis, DocumentScore, etc.)
└── clause_analyzer.py  # LLM per-clause analysis + document-level analysis
```
