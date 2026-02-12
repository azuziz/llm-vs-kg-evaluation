# Prompts

This folder contains the prompt variants used to query the LLM. Each prompt file
represents an experimental hypothesis about how instructions change model behavior.

All prompt variants are used together with **Structured Outputs** (JSON schema),
where the schema enforces:
- output must be valid JSON
- `predicate` must be one of the allowed Biolink predicates (enum)
- `subject` and `object` must be the provided CURIEs
- no additional properties

Because the schema already constrains the output strongly, prompt differences focus
on *semantic calibration* (conservative vs forced prediction) and *role/directionality
guidance*.

## Files

### `prompt_v1_minimal.txt`
**Minimal instruction set**.
Goal:Extract any biomedical relations that reasonably make sense between the two nodes.

### `prompt_v2_curated_kb.txt`
Adds language like “commonly asserted in curated biomedical knowledge bases”.
Goal: push the model toward **conservative, KG-like assertions** and reduce speculative 
edges.
Expected behavior: fewer edges, higher precision-like overlap, potentially lower recall.

### `prompt_v3_directional.txt`
Extract only relations that are widely documented in authoritative biomedical knowledge 
bases (e.g., curated ontologies, databases).Goal: reduce subject/object flipping and 
inverse predicate usage.
Expected behavior: requires stronger evidence than mere plausibility, multiple 
relations allowed, if none are commonly asserted → empty edges list.

### `prompt_v4_forced_choice.txt`
Forces at least one edge (i.e., discourages empty output).
Goal: test whether the “empty edges allowed” instruction is driving false negatives.
Expected behavior: higher recall-like scores but inflated false positives / lower precision.
