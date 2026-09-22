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
| `prompt_v2_most_plausible.txt` | v2, single most plausible relation | Return the single most plausible relation based on general biomedical knowledge, only if at least one relation is plausibly supported; otherwise an empty edges list. |
| `prompt_v3_curated_kb.txt` | v3, curated KB constraint | Return only relations that are commonly asserted in curated biomedical knowledge bases; if none is commonly asserted, an empty edges list. Multiple relations are allowed. |
| `prompt_v4_most_asserted.txt` | v4, single most commonly asserted curated relation | Return the single most commonly asserted relation in curated biomedical knowledge bases, if any such assertion exists; otherwise an empty edges list. |

File names were renamed to match their content (v2 was previously named `prompt_v2_curated_kb.txt`, which is actually the curated-KB constraint of v3, and v3 was named `prompt_v3_directional.txt`; v4 was named `prompt_v4_forced_choice.txt`, though it allows an empty edges list and so does not force a non-empty answer). The historical copy under `project_history/10_prompt_design/prompts/` keeps the original file names, matching what the scripts in that folder were actually run against.

## Repetition budget

The baseline v1 was executed 100 times per question (16,300 runs). Variants v2–v4 were executed 25 times per question (4,075 runs per variant) as a prompt-sensitivity study. Results per variant (agreement, output volume, abstention) are reported in Chapter 4 of the thesis (Section 4.5.2).
