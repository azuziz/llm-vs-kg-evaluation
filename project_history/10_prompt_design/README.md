# Prompt Design Experiment: GPT vs ROBOKOP

## 1. Motivation

In previous experiments, each biomedical node pair was queried 100 times using an identical
prompt, with outputs constrained to a fixed set of allowed predicates.
Observed variability raised a key question:

**Is output variance driven primarily by stochastic sampling, or by prompt formulation itself?**

This experiment isolates **prompt-induced variance** by holding all other factors constant.


---

## 2. Experimental Question

> How much does prompt wording influence biomedical relation extraction quality,
> compared to repeated sampling with the same prompt?

Specifically:
- Are multiple runs with the same prompt redundant?
- Do different prompts systematically shift alignment with ROBOKOP?


---

## 3. Experimental Design

### Fixed parameters (invariants)

The following are identical across all runs:

- Model: `gpt-5.2`
- Temperature
- Max output tokens
- Allowed predicate list (≈75 Biolink predicates)
- Structured JSON schema (enum-enforced predicates)
- Canonicalization and post-processing logic
- Evaluation metrics
- Node pairs (163)
- ROBOKOP reference knowledge graph

### Variable

Only **prompt text** changes.

### Prompt variants

| ID | Prompt focus |
|----|-------------|
| v1 | Minimal baseline |
| v2 | Curated knowledge base framing |
| v3 | Explicit directionality constraint |
| v4 | Forced decision (edge vs no edge) |

### Sampling

- 25 independent runs per prompt variant
- 163 node pairs per run

Total GPT calls:
4 × 25 × 163 = 16,300
---

## 4. Data Flow

node_pairs + prompt_variant
↓
GPT structured calls
↓
raw per-run outputs
↓
per-prompt aggregation
↓
ROBOKOP comparison
↓
cross-prompt analysis

---

## 5. Outputs

### Per-run
- Raw GPT edges per prompt variant

### Per-prompt (aggregated)
- Yes / no-edge rate
- Predicate stability
- Edge count distribution

### Evaluation
- Jaccard similarity
- Precision-like containment
- Recall-like containment
- F1 score

All metrics are computed:
- per run
- per question
- per prompt variant

---

## 6. Interpretation Goals

This experiment allows us to:

- Quantify prompt-induced variance
- Compare prompt variance to sampling variance
- Identify the most stable and best-aligned prompt
- Decide whether future experiments should prioritize
  - more runs, or
  - better prompts

---

## 7. Expected Outcomes

- If prompt variance ≈ sampling variance → reduce runs
- If prompt variance ≫ sampling variance → prompt choice dominates
- If one prompt consistently dominates → use as canonical prompt

---

## 8. Reproducibility

All prompt texts, configs, and scripts are versioned in this directory.
No external state is required beyond the referenced data files.
