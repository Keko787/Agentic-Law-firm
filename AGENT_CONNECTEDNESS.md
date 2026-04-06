# Agent Connectedness — How the Pipeline Fits Together

This document explains how the 5 agents in the Agentic Lawyer pipeline pass data to each other, with exact code references showing where each handoff occurs.

---

## Pipeline Data Flow

```
main.py line 58:   intake_result    = intake_agent.process(pdf_path)
                        |
                        v
main.py line 70:   research_result  = research_agent.research(intake_result)
                        |                   |
                        v                   v
main.py line 85:   analysis_result  = analysis_agent.analyze(intake_result, research_result)
                        |                   |
                        v                   v
main.py line 97:   evaluation_result = evaluator_agent.evaluate(analysis_result, research_result)
                        |                   |
                        v                   v
main.py line 117:  final_report      = output_agent.generate_report(analysis_result, evaluation_result)
```

Every arrow is a typed Pydantic model. Every intermediate result is saved as JSON in the `output/` directory, making any stage independently debuggable.

---

## Handoff 1: Intake Agent → Research Agent

**Where:** `main.py` lines 58 and 70

```python
intake_result = intake_agent.process(pdf_path)                          # Stage 1
research_result = research_agent.research(intake_result)                # Stage 2
```

**What gets passed:** `IntakeResult` — contains document metadata (type, parties, jurisdiction) and a list of classified `Clause` objects (each with type, risk level, text, flags).

**How Research uses it:** Inside `research_agent/agent.py` (lines 103-108), the Research Agent reads the jurisdiction for filtering and iterates through the clauses sorted by risk:

```python
jurisdiction = intake_result.metadata.jurisdiction or ""
doc_type = intake_result.metadata.document_type
clauses = self._prioritize_clauses(intake_result.clauses, risk_threshold)
```

The `_prioritize_clauses` method (line 217-235) filters out preamble/signature blocks and sorts by risk level so high-risk clauses are researched first. The clause type and risk level drive query generation — a high-risk non-compete clause gets different search queries than a low-risk definitions clause.

---

## Handoff 2: Intake + Research → Analysis Agent

**Where:** `main.py` line 85

```python
analysis_result = analysis_agent.analyze(intake_result, research_result)
```

**What gets passed:** Both `IntakeResult` and `ResearchResult`. This is the three-way merge — the core value of the multi-agent architecture.

**How Analysis uses it:** Inside `analysis_agent/agent.py` (lines 80-101), the Analysis Agent matches research to clauses by index:

```python
research_by_index = {cr.clause_index: cr for cr in research_result.clause_research}

for clause in intake_result.clauses:
    if clause.clause_type.value in ("preamble", "signature_block"):
        continue
    research = research_by_index.get(clause.index)
    analysis = self._analyze_single_clause(clause, research)
```

The `_analyze_single_clause` method (lines 151-172) combines data from both agents into a single LLM call:

```python
return self.analyzer.analyze_clause(
    clause_text=clause.text,                                    # from Intake
    clause_heading=clause.heading,                              # from Intake
    clause_type=clause.clause_type.value,                       # from Intake
    risk_level=clause.risk_level.value,                         # from Intake
    intake_flags=clause.flags,                                  # from Intake
    intake_summary=clause.summary,                              # from Intake
    research_context=research.legal_context if research else "",            # from Research
    research_risk_analysis=research.risk_analysis if research else "",      # from Research
    research_recommendations=research.recommendations if research else [],  # from Research
    research_enforceability=research.enforceability_notes if research else "", # from Research
    applicable_law=research.applicable_law if research else [],             # from Research
)
```

This is the three-way merge in action: clause text + intake classification + research synthesis all feed into a single LLM prompt that produces grounded findings citing actual retrieved sources.

---

## Handoff 3: Analysis + Research → Evaluator Agent

**Where:** `main.py` line 97

```python
evaluation_result = evaluator_agent.evaluate(analysis_result, research_result)
```

**What gets passed:** `AnalysisResult` (the findings to validate) and `ResearchResult` (the sources to validate against). The Evaluator needs both because its job is to check whether the Analysis Agent's findings are actually grounded in the research that was available.

**How Evaluator uses it:** Inside `evaluator_agent/agent.py` (lines 127-143), it matches research back to each clause analysis:

```python
research_by_index = {cr.clause_index: cr for cr in research_result.clause_research}

for ca in analysis_result.clause_analyses:
    research = research_by_index.get(ca.clause_index)
    eval_result = self._evaluate_clause(ca, research)
```

The `_evaluate_clause` method (lines 196-262) sends each finding alongside the available research to an LLM reviewer. The prompt asks whether:
- The legal basis is supported by the retrieved sources
- The severity is proportionate
- The suggested revision is actionable
- There are any consistency issues

This cross-referencing is what catches hallucinations — if the Analysis Agent cited a statute that wasn't in the retrieved sources, the Evaluator flags it.

---

## Handoff 4: Reflection Loop (Evaluator → Analysis)

**Where:** `main.py` lines 104-109

```python
if evaluation_result.clauses_needing_reanalysis:
    print(f"Reflection loop: {len(evaluation_result.clauses_needing_reanalysis)} "
          f"clause(s) flagged for reanalysis")
    # In a full implementation, we would re-run the analysis agent
    # on flagged clauses with additional context from the evaluator.
    print("   (Reanalysis noted — proceeding with current results)")
```

**Current state:** The Evaluator correctly identifies which clauses need re-analysis (inside `evaluator_agent/agent.py` lines 160-163):

```python
if any(ce.confidence == ConfidenceLevel.FAILED for ce in clause_evals):
    overall = ConfidenceLevel.LOW
elif len(reanalysis_needed) > 0:
    overall = ConfidenceLevel.MEDIUM
```

The trigger condition is: any finding scores `failed` OR 2+ findings score `low`. The clause indices are stored in `evaluation_result.clauses_needing_reanalysis`. The re-analysis execution path (sending flagged clauses back through the Analysis Agent with evaluator feedback) is wired in the data model but the retry loop is not yet implemented in `main.py`.

---

## Handoff 5: Analysis + Evaluation → Output Agent

**Where:** `main.py` line 117

```python
final_report = output_agent.generate_report(analysis_result, evaluation_result)
```

**What gets passed:** `AnalysisResult` (findings, scores, missing clauses) and `EvaluationResult` (confidence, hallucination flags, pass rate).

**How Output uses it:** Inside `output_agent/agent.py`, different sections pull from different sources:

From `analysis_result`:
- Executive summary (line 159): risk score, top concerns, critical findings
- Risk scorecard (line 201): document score, findings breakdown
- Clause-by-clause findings (lines 96-101): sorted by severity, with legal basis and revisions
- Missing clauses (line 104): suggested language
- Cross-clause issues (line 108): spanning issues
- Negotiation playbook (line 112): critical/major findings with revision language

From `evaluation_result`:
- Executive summary (line 176): overall confidence level
- Clause sections (line 101): evaluation confidence per clause
- Confidence disclosure (lines 310-330): pass rate, score validation, hallucination flags, legal disclaimer

The Output Agent also generates the executive summary via a separate LLM call (lines 151-199), passing it a digest of the analysis and evaluation data to produce plain-language text for non-lawyers.

---

## Why This Architecture Works

The key insight is that **each agent adds something the others cannot**:

| Agent | Unique contribution |
|-------|-------------------|
| **Intake** | PDF parsing + rule-based segmentation (no LLM needed for 80% of structure detection) |
| **Research** | RAG retrieval from legal corpus (grounds everything in actual legal authority) |
| **Analysis** | Three-way merge (clause + classification + research = findings that cite real sources) |
| **Evaluator** | Cross-reference validation (catches hallucinations by comparing findings to sources) |
| **Output** | Plain-language translation (turns structured analysis into a client-ready report) |

No single agent has access to everything. The Research Agent doesn't know the Analysis Agent's findings. The Evaluator doesn't re-read the PDF. The Output Agent doesn't query the vector store. Each agent operates on the structured outputs of its predecessors, which makes the pipeline:

- **Debuggable** — inspect any intermediate JSON to find where something went wrong
- **Testable** — mock any stage's output to test downstream agents in isolation
- **Extensible** — add a new agent or modify one without breaking others
- **Auditable** — every finding can be traced back through research to its source in the corpus

---

## Typed Contracts Between Agents

All handoffs use Pydantic models with JSON serialization:

```
IntakeResult      → research_agent.research()
                  → analysis_agent.analyze()

ResearchResult    → analysis_agent.analyze()
                  → evaluator_agent.evaluate()

AnalysisResult    → evaluator_agent.evaluate()
                  → output_agent.generate_report()

EvaluationResult  → output_agent.generate_report()

FinalReport       → saved as JSON + Markdown
```

Each model defines exactly what fields downstream agents can rely on. Adding a field to `IntakeResult` doesn't break the Research Agent. Changing the Analysis Agent's prompt doesn't break the Output Agent. The typed boundaries make the system modular.
