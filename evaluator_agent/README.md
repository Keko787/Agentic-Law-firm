# Evaluator Agent (Stage 4)

The Evaluator Agent is the quality gate of the Agentic Lawyer pipeline. It validates the Analysis Agent's output before it reaches the user by checking that findings are grounded in retrieved legal sources, severities are proportionate, and no citations are hallucinated.

This is the key differentiator from single-agent systems: the Evaluator catches errors that inline checks would miss.

## What It Does

```
AnalysisResult + ResearchResult
    |
    v
+-----------------------------------------+
| Per-clause evaluation:                  |
|   For each finding:                     |
|     - Is it grounded in research?       |
|     - Is severity proportionate?        |
|     - Is the revision actionable?       |
|     - Are there consistency issues?     |
|   --> Confidence: high/medium/low/failed|
+-----------------------------------------+
    |
    v
+-----------------------------------------+
| Document-level validation:              |
|   - Is risk score consistent?           |
|   - Any hallucinated citations?         |
|   - Do cross-clause issues hold up?     |
+-----------------------------------------+
    |
    v
+-----------------------------------------+
| Reflection loop trigger:                |
|   If finding = "failed" OR 2+ = "low"  |
|   --> Flag clause for re-analysis       |
+-----------------------------------------+
    |
    v
EvaluationResult
```

## Two-Pass Evaluation

### Pass 1: Per-Clause Evaluation

For each clause in the `AnalysisResult`, the Evaluator sends the clause's findings alongside the research that was available to an LLM reviewer. For each finding, it checks:

- **Grounding** — Is the legal basis actually supported by retrieved sources?
- **Severity** — Is the severity proportionate to the actual risk?
- **Actionability** — Is the suggested revision concrete enough to use?
- **Consistency** — Do findings contradict each other or the research?

Each finding receives a confidence score:
| Level | Meaning |
|-------|---------|
| **HIGH** | Directly supported by retrieved sources, legal basis is accurate, severity is proportionate, revision is concrete |
| **MEDIUM** | Reasonable but legal basis is partially supported or generic, severity could be debated |
| **LOW** | Lacks clear support in retrieved sources, legal basis appears fabricated or misapplied |
| **FAILED** | Contradicts the research, cites non-existent law, or contains obvious errors |

### Pass 2: Document-Level Validation

Validates the overall analysis:
- Is the risk score consistent with the actual findings?
- Are there hallucinated citations across any clauses?
- Do the cross-clause issues identified by the Analysis Agent hold up?

## Reflection Loop

The Evaluator triggers re-analysis when quality is insufficient:

- **Trigger condition:** Any finding scores `failed` OR 2+ findings score `low`
- **Action:** The clause index is added to `clauses_needing_reanalysis`
- **Current state:** The flag is set and logged; the re-analysis execution path is wired in the data model but the retry loop in `main.py` is not yet implemented

## Key Metric

**`pass_rate`** — the percentage of clauses achieving `high` or `medium` confidence. This is the primary quality indicator for the pipeline output.

## Data Models (`models.py`)

| Model | Description |
|-------|-------------|
| `ConfidenceLevel` | Enum: `high`, `medium`, `low`, `failed` |
| `FindingEvaluation` | Evaluation of a single finding: is_grounded, confidence, grounding notes, issues |
| `ClauseEvaluation` | Evaluation of a clause: confidence, finding evaluations, consistency issues, needs_reanalysis flag |
| `EvaluationResult` | Full output: overall confidence, clause evaluations, score validation, hallucination flags, clauses needing re-analysis |

## Usage

```python
from evaluator_agent import EvaluatorAgent

agent = EvaluatorAgent(api_key="sk-ant-...")
eval_result = agent.evaluate(analysis_result, research_result)
eval_result.save("output/evaluation_result.json")

# Check overall quality
print(f"Confidence: {eval_result.overall_confidence.value}")
print(f"Pass rate: {eval_result.pass_rate:.0%}")

# Check for hallucinations
if eval_result.hallucination_flags:
    print("Hallucination flags:")
    for flag in eval_result.hallucination_flags:
        print(f"  - {flag}")

# Check which clauses need re-analysis
if eval_result.clauses_needing_reanalysis:
    print(f"Clauses needing re-analysis: {eval_result.clauses_needing_reanalysis}")
```

## Output

The `EvaluationResult` is passed to the Output Agent (Stage 5), which includes confidence scores and grounding warnings in the final report's "Confidence and disclaimers" section.

## Files

```
evaluator_agent/
├── __init__.py
├── agent.py    # EvaluatorAgent orchestrator (evaluate method)
└── models.py   # Pydantic data models (FindingEvaluation, ClauseEvaluation, EvaluationResult)
```
