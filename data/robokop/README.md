# ROBOKOP reference triples (KG baseline)

This folder contains the **reference edges** obtained by querying ROBOKOP for each
node pair in `data/node_pairs/node_pairs_163.tsv`.

These edges serve as a **KG-derived reference set** for evaluation. They are not
assumed to be complete biological truth; they represent what is explicitly asserted
in ROBOKOP’s integrated knowledge sources at the time of retrieval.

## Files

### `robokop_triples.tsv`
Tab-separated table of ROBOKOP edges collected for each question (node pair).

Each row corresponds to one KG edge returned by ROBOKOP.

Columns:

- `question_id`  
  Links the edge back to the originating node pair (e.g., `Q1`).

- `query_edge_key`  
  Identifier for the query edge in the ROBOKOP query template (e.g., `e0`).
  (Useful when multi-edge TRAPI queries are used; here primarily for provenance.)

- `edge_id`  
  Local edge identifier assigned during extraction (e.g., `e_1`, `e_2`).

- `subject`, `object`  
  CURIEs of the connected nodes.

- `predicate`  
  Biolink predicate (e.g., `biolink:genetically_associated_with`).

- `subject_label`, `object_label`  
  Human-readable labels for debugging and reporting.

- `provenance`  
  Source attribution strings. Example:  
  `infores:hetionet(primary_knowledge_source);infores:automat-robokop(aggregator_knowledge_source)`

  These indicate:
  - primary knowledge sources contributing the assertion
  - ROBOKOP/Automat as aggregator

## How it is used in evaluation

For each `question_id`, the set of ROBOKOP edges is transformed into a canonical
representation (typically the triple `(subject, predicate, object)`), and compared
to the set of LLM-produced edges for the same question.

Common evaluation metrics computed downstream include:
- Jaccard overlap
- precision-like containment
- recall-like containment
- F1-like harmonic mean

## Important caveats

- ROBOKOP edges are **directional** and depend on the KG modeling conventions.
  LLM outputs may be biologically plausible but directionally inverted relative to
  the KG; this is a known evaluation challenge.
- The ROBOKOP KG is updated over time. This file is a **snapshot** that fixes the
  reference set for reproducible experiments.
