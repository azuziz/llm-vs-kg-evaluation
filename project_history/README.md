# Project history

The stage folders below are the working directories of the thesis, kept in the order they were worked on (numbering = chronology). The curated, reproducible core of the project (prompts, the 163 node pairs, ROBOKOP triples) lives in the repository root (`data/`, `prompts/`, `environment/`); the final thesis is in `thesis/BachelorArbeit.pdf`.

These folders are kept as-is, including exploratory scripts and superseded results, so the path from first API experiments to the final results stays traceable. Some scripts contain absolute paths from the original machine.

| Stage | Folder | Content |
|---|---|---|
| 0 | `0_gpt_api` | First GPT API request scripts and responses |
| 1 | `1_gpt_robokop_eval` | First GPT-vs-ROBOKOP triple extraction and comparison scripts |
| 2 | `2_gpt_feed_predicates_CURIE` | Placeholder (empty) |
| 3 | `3_gpt_translateTheQuestions` | Translating questions into TRAPI queries |
| 4 | `4_gpt` | ROBOKOP OpenAPI predicate extraction, per-question KG data |
| 5 | `5_gpt` | First trial: GPT raw responses, ROBOKOP query and triple extraction |
| 6 | `6_gpt` | Multi-step pipeline with allowed predicates and confusion-matrix metrics |
| 7 | `7_gpt` | Cleaned evaluation pipeline (collect, metrics, plots) |
| 8 | `8_questions` | Question / node-pair selection (163 questions) and repeated GPT runs |
| 9 | `9_visualisation` | Plotting of per-run metrics |
| 10 | `10_prompt_design` | Prompt variants v1-v4: collect, canonicalize, compare, plot |
| 11 | `11_redo` | Re-run and per-question edge / predicate statistics |
| 12 | `12_stats` | ROBOKOP deduplication and KG edge-count statistics |
| 13 | `13_dedup_statistics` | Deduplication statistics pipeline for the thesis (section 4.1) |
| 14 | `14_united_results` | Union evaluation, stability analysis and key-metric plots across prompt versions |
| - | `statistics` | Early exploratory statistics and plots |

## Not included

Python virtual environments, IDE settings and files containing API keys were deliberately left out.
