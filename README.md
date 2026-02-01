# \# Evaluation of LLMs for Biomedical Relation Discovery

# 

# \## Motivation

# 

# Large Language Models (LLMs) are increasingly used to answer biomedical questions and to propose relations between biological entities such as genes, diseases, and drugs. However, evaluating the quality of such relations is non-trivial. Curated biomedical Knowledge Graphs (KGs), such as ROBOKOP, represent expert-curated, conservative, and incomplete snapshots of biomedical knowledge, whereas LLM outputs are generative, probabilistic, and inherently variable across runs. Directly comparing LLM-generated relations to KG assertions therefore raises fundamental methodological questions regarding ground truth, directionality, predicate choice, and reproducibility. This project investigates how LLM-generated biomedical relations align with curated KG relations under controlled constraints, rather than assuming exact equivalence between the two.

# 

# \## Research Question

# 

# This project evaluates the extent to which LLMs can reproduce \*plausible, curated biomedical relations\* between fixed pairs of entities when constrained to a predefined predicate space. Specifically, it asks: \*\*How closely do LLM-generated relations overlap with relations asserted in a curated biomedical knowledge graph when evaluated using set-based similarity metrics across repeated runs?\*\*

# 

# This work does \*\*not\*\* aim to validate novel biological discoveries, establish causal relationships, or assess biological correctness beyond alignment with curated knowledge. The KG is treated as a reference comparator rather than an absolute or complete ground truth. The focus is on overlap, variability, and consistency of relation discovery under repeated sampling.

# 

# \## Method Overview

# 

# \- \*\*Node pairs\*\*  

# &nbsp; A fixed set of biomedical entity pairs (e.g. disease–gene, gene–drug) is selected. Each pair defines a single evaluation question and is reused across all LLM runs to ensure comparability.

# 

# \- \*\*ROBOKOP reference\*\*  

# &nbsp; For each node pair, all corresponding relations present in the ROBOKOP knowledge graph are extracted and used as a curated reference set.

# 

# \- \*\*LLM prompting\*\*  

# &nbsp; The LLM is prompted to extract relations between exactly the two given nodes, using a controlled natural-language prompt. Multiple independent runs are performed per node pair to capture output variability.

# 

# \- \*\*Structured outputs\*\*  

# &nbsp; LLM responses are constrained via a JSON schema that enforces a fixed set of allowed predicates and explicit subject–predicate–object triples, ensuring syntactic validity and reducing prompt-induced variability.

# 

# \- \*\*Metrics\*\*  

# &nbsp; LLM-generated relation sets are compared to ROBOKOP reference sets using set-based metrics, including Jaccard similarity, precision-like containment, recall-like containment, and F1 score. Metrics are analyzed per question and across repeated runs to quantify stability and overlap.

# 

# \## Repository Structure

# 

# The repository is organized into data, scripts, outputs, and documentation. Input data (node pairs, allowed predicates, and ROBOKOP triples) are stored in `data/`. All Python scripts for LLM querying, aggregation, comparison, and plotting are located in `scripts/`. Generated outputs, including raw LLM runs, aggregated metrics, and figures, are written to `out/`. Additional documentation, design notes, and experiment logs are kept in `docs/`.

# 

# \## Status

# 

# The core pipeline for LLM querying, structured output validation, relation aggregation, and metric computation is implemented and reproducible. Initial experiments with repeated LLM runs per node pair have been completed, and variability-aware metrics have been computed. Ongoing work includes refining directionality handling, exploring predicate pairing and inverse predicates, and extending the analysis of stability and disagreement across runs.



