# Prompts

This folder contains the four prompt variants (v1–v4) used to query the LLM. The variants differ only in a small number of instruction-framing and evidence-requirement constraints, so that prompt sensitivity can be compared on the identical set of 163 node pairs. The exact texts are the `.txt` files in this folder; they are also reproduced verbatim in the thesis (Appendix A.1).

All prompt variants are used together with **Structured Outputs** (JSON schema),
where the schema enforces:
- output must be valid JSON
- `predicate` must be one of the allowed Biolink predicates (enum)
- `subject` and `object` must be the provided CURIEs
- no additional properties
- the `edges` list may contain zero or more triples (an empty list is a valid output)

Because the schema already constrains the output strongly, prompt differences focus
on *how strong an evidence requirement* the instruction places on a returned relation.

## Files

| File | Variant (as named in the thesis) | Instruction in short |
|---|---|---|
| `prompt_v1_minimal.txt` | v1, baseline | Return only relations that plausibly hold. |
| `prompt_v2_curated_kb.txt` | v2, single most plausible relation | Return the single most plausible relation based on general biomedical knowledge, only if at least one relation is plausibly supported; otherwise an empty edges list. |
| `prompt_v3_directional.txt` | v3, curated KB constraint | Return only relations that are commonly asserted in curated biomedical knowledge bases; if none is commonly asserted, an empty edges list. Multiple relations are allowed. |
| `prompt_v4_forced_choice.txt` | v4, single most commonly asserted curated relation | Return the single most commonly asserted relation in curated biomedical knowledge bases, if any such assertion exists; otherwise an empty edges list. |

**Note on file names.** The file names are historical working names and do not always describe the final prompt. In particular, `prompt_v2_curated_kb.txt` is the "single most plausible relation" prompt and `prompt_v3_directional.txt` is the "curated KB constraint" prompt, and neither v3 nor v4 forces a non-empty answer. The variant labels v1–v4 and the descriptions in the thesis are authoritative.

## Repetition budget

The baseline v1 was executed 100 times per question (16,300 runs). Variants v2–v4 were executed 25 times per question (4,075 runs per variant) as a prompt-sensitivity study. Results per variant (agreement, output volume, abstention) are reported in Chapter 4 of the thesis (Section 4.5.2).
