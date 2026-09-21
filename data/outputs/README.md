# Outputs (LLM runs, aggregated results, and metrics)

This folder is reserved for storing **generated outputs** from LLM runs and/or
post-processed derived tables.

The goal is to keep:

- raw run-level outputs (for reproducibility and error analysis)
- aggregated per-question outputs (for evaluation)
- metric tables (for plotting and reporting)

> **Where the outputs are in this repository:** the generated outputs of the
> thesis experiments are kept in [`project_history/`](../../project_history/README.md),
> next to the scripts that produced them. The final prompt-variant runs and
> metrics are in `project_history/10_prompt_design/out/` and
> `project_history/14_united_results/`. The files described below first appear in
> `project_history/6_gpt/` (`gpt_runs.csv`, `gpt_aggregated.csv`,
> `step3_per_question.csv`); `step3_per_run_metrics.csv` is in
> `project_history/8_questions/thosecomputations/` and `project_history/9_visualisation/out/`.
> This folder itself only documents the conventions.

## Contents (conventions)

Depending on pipeline stage, typical files are:

- `gpt_runs.csv`
  One row per (question_id, run_id), containing the raw structured JSON response
  or a normalized edge list.

- `gpt_aggregated.csv`
  Per-question aggregation of repeated runs (e.g., stability measures, mode output,
  invalid rate).

- `step3_per_run_metrics.csv`
  Per-run comparison metrics against ROBOKOP (Jaccard, precision-like, recall-like, F1).

- `step3_per_question.csv`
  Metrics summarized per question (means, medians, variances, etc.).

## The role of this folder

Outputs evolve during development and experimentation. Keeping them under version
control (or at least keeping *schemas and small samples* under version control)
allows to:

- reproduce plots exactly
- debug failures (schema invalidity, retry artifacts)
- compare prompt variants on identical inputs

## Size warning

Raw outputs can become large (e.g., 163 questions × 100 runs). If files exceed
reasonable repo size, store only:

- small samples + schemas + metadata in GitHub

and keep the full outputs externally (e.g., local disk),
referenced by an experiment log.

For this repository the outputs were small enough (about 70 MB in total) that the
complete set of runs was kept under version control in `project_history/`.
