\# Outputs (LLM runs, aggregated results, and metrics)



This folder is reserved for storing \*\*generated outputs\*\* from LLM runs and/or

post-processed derived tables.



The goal is to keep:

\- raw run-level outputs (for reproducibility and error analysis)

\- aggregated per-question outputs (for evaluation)

\- metric tables (for plotting and reporting)



\## Contents (conventions)



Depending on pipeline stage, typical files are:



\- `gpt\_runs.csv`  

&nbsp; One row per (question\_id, run\_id), containing the raw structured JSON response

&nbsp; or a normalized edge list.



\- `gpt\_aggregated.csv`  

&nbsp; Per-question aggregation of repeated runs (e.g., stability measures, mode output,

&nbsp; invalid rate).



\- `step3\_per\_run\_metrics.csv`  

&nbsp; Per-run comparison metrics against ROBOKOP (Jaccard, precision-like, recall-like, F1).



\- `step3\_per\_question.csv`  

&nbsp; Metrics summarized per question (means, medians, variances, etc.).



\## The role of this folder



Outputs evolve during development and experimentation. Keeping them under version

control (or at least keeping \*schemas and small samples\* under version control)

allows to:

\- reproduce plots exactly

\- debug failures (schema invalidity, retry artifacts)

\- compare prompt variants on identical inputs



\## Size warning



Raw outputs can become large (e.g., 163 questions × 100 runs). If files exceed

reasonable repo size, store only:

\- small samples + schemas + metadata in GitHub

and keep the full outputs externally (e.g., local disk),

referenced by an experiment log.



