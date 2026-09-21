# Evaluation of LLMs for Biomedical Relation Discovery

Bachelor thesis project. The full write-up is in [`thesis/BachelorArbeit.pdf`](thesis/BachelorArbeit.pdf).

## Motivation

Large Language Models (LLMs) are increasingly used to answer biomedical questions and to propose relations between biological entities such as genes, diseases, and drugs. However, evaluating the quality of such relations is non-trivial. Curated biomedical Knowledge Graphs (KGs), such as ROBOKOP, represent expert-curated, conservative, and incomplete snapshots of biomedical knowledge, whereas LLM outputs are generative, probabilistic, and inherently variable across runs. Directly comparing LLM-generated relations to KG assertions therefore raises fundamental methodological questions regarding ground truth, directionality, predicate choice, and reproducibility. This project investigates how LLM-generated biomedical relations align with curated KG relations under controlled constraints, rather than assuming exact equivalence between the two.

## Research Question

This project evaluates the extent to which LLMs can reproduce *plausible, curated biomedical relations* between fixed pairs of entities when constrained to a predefined predicate space. Specifically, it asks: **How closely do LLM-generated relations overlap with relations asserted in a curated biomedical knowledge graph when evaluated using set-based similarity metrics across repeated runs?**

This work does **not** aim to validate novel biological discoveries, establish causal relationships, or assess biological correctness beyond alignment with curated knowledge. The KG is treated as a reference comparator rather than an absolute or complete ground truth. The focus is on overlap, variability, and consistency of relation discovery under repeated sampling.

## Method Overview

- **Node pairs**
  A fixed set of biomedical entity pairs (e.g. disease–gene, gene–drug) is selected. Each pair defines a single evaluation question and is reused across all LLM runs to ensure comparability.

- **ROBOKOP reference**
  For each node pair, all corresponding relations present in the ROBOKOP knowledge graph are extracted and used as a curated reference set.

- **LLM prompting**
  The LLM is prompted to extract relations between exactly the two given nodes, using a controlled natural-language prompt. Multiple independent runs are performed per node pair to capture output variability.

- **Structured outputs**
  LLM responses are constrained via a JSON schema that enforces a fixed set of allowed predicates and explicit subject–predicate–object triples, ensuring syntactic validity and reducing prompt-induced variability.

- **Metrics**
  LLM-generated relation sets are compared to ROBOKOP reference sets using set-based metrics, including Jaccard similarity, precision-like containment, recall-like containment, and F1 score. Metrics are analyzed per question and across repeated runs to quantify stability and overlap.

## Repository Structure

| Path | Content |
|---|---|
| `data/node_pairs/` | The 163 evaluation node pairs (`node_pairs_163.tsv`) |
| `data/robokop/` | ROBOKOP reference triples for these pairs (`robokop_triples.tsv`) |
| `data/outputs/` | Conventions for generated outputs (raw runs, aggregates, metrics); the outputs themselves are in `project_history/` |
| `prompts/` | The four prompt variants: v1 minimal, v2 curated KB, v3 directional, v4 forced choice |
| `environment/` | Experiment configuration (`experiment.yaml`) |
| `project_history/` | Complete working history of the thesis, one folder per stage (numbered in the order they were worked on), containing all scripts, raw LLM runs, metrics and plots. See [`project_history/README.md`](project_history/README.md) |
| `thesis/` | The bachelor thesis as PDF |

Where to find the code in `project_history/`:

- `10_prompt_design/`: prompt-variant pipeline (collect runs, canonicalize, compare against ROBOKOP, plot)
- `12_stats/` and `13_dedup_statistics/`: ROBOKOP deduplication and KG edge-count statistics
- `14_united_results/`: union evaluation, stability analysis and key metrics across prompt versions
- `0_` to `9_`: earlier stages, from the first API experiments to the first full evaluation pipeline

The allowed-predicate list used for the JSON schema is at `project_history/7_gpt/data/allowed_predicates.json`. Python virtual environments, IDE files and API keys are not part of the repository; the scripts expect an OpenAI API key in the `OPENAI_API_KEY` environment variable.

## Status

The project is finished and documented in the bachelor thesis. The pipeline for LLM querying, structured output validation, relation aggregation, and metric computation is implemented, repeated LLM runs per node pair were completed for the prompt variants, and variability-aware metrics were computed and analyzed.
