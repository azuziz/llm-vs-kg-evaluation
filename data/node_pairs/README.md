# Node pairs (input questions)

This folder contains the curated **node-pair dataset** used as input to both:
1) ROBOKOP querying (to obtain reference KG edges), and  
2) LLM prompting (to generate predicted edges under constrained predicates).

The key design choice is that **all node pairs have at least one ROBOKOP edge** (typically 1–10) so that evaluation is not dominated by “true negative / empty KG” ambiguity.

## Files

### `node_pairs_163.tsv`
Tab-separated file defining the evaluation questions. Each row is one **question** (Q1..Q163) consisting of exactly two nodes.

Columns:

- `question_id`  
  Unique identifier (e.g., `Q1`) used to join outputs across the whole pipeline.

- `subject_text`, `object_text`  
  Human-readable strings used during dataset construction / sanity checks.

- `subject_curie`, `object_curie`  
  Canonical CURIE identifiers (e.g., `MONDO:0005180`, `NCBIGene:348`).  
  **These are the only identifiers allowed to appear in model output.**

- `subject_label`, `object_label`  
  Human-readable labels (often the same as text); included for readability.

- `notes`  
  Compact metadata string. Example:
  `SUBJ:DISEASE_MONDO|OBJ:GENE_NCBI|ROBOKOP_EDGES=2`

  Intended meaning:
  - subject/object high-level type source (e.g. MONDO disease, NCBI gene)
  - `ROBOKOP_EDGES=n` is the number of edges found in ROBOKOP for that pair
    at the time the dataset was generated.

## Intended usage

- The `*_curie` fields are used to:
  - query ROBOKOP
  - constrain the LLM structured output to only these two node IDs

- `question_id` is the join key across:
  - ROBOKOP triples (`data/robokop/robokop_triples.tsv`)
  - LLM outputs (stored under `data/outputs/` or `experiments/`)
  - metrics tables and plots

## Notes / limitations

- This dataset is **not** a random sample of biomedical pairs. It is curated to
  avoid pairs with zero KG evidence.
- The number of ROBOKOP edges is sensitive to ROBOKOP/KG updates; it is recorded
  only as metadata for the dataset generation snapshot.
