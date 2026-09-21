import csv
import json
import re
import sys
import time
import requests

NAME_RESOLVER = "https://name-resolution-sri.renci.org/lookup"
NODE_NORM = "https://nodenormalization-sri.renci.org/get_normalized_nodes"

# HGNC gene symbol -> HGNC numeric id via ClinicalTables HGNC gene API
# Docs: https://clinicaltables.nlm.nih.gov/apidoc/genes/v3/doc.html
HGNC_CT_API = "https://clinicaltables.nlm.nih.gov/api/genes/v3/search"

def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def strip_parens(s: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*", "", s).strip()

def norm_gene_symbol(s: str) -> str:
    return strip_parens(s).upper()

def is_mutation_phrase(text: str) -> bool:
    t = text.lower()
    return ("mutation" in t) or ("mutations" in t) or ("variant" in t)

def is_gene_like(text: str) -> bool:
    t = norm_gene_symbol(text)
    # gene symbols are typically short uppercase alnum (TP53, FOXO1, etc.)
    return bool(re.fullmatch(r"[A-Z0-9]{2,12}", t))

def node_normalize(curies):
    r = requests.post(NODE_NORM, json={"curies": curies}, timeout=60)
    r.raise_for_status()
    return r.json()

def canonicalize(curie: str):
    nn = node_normalize([curie])
    rec = nn.get(curie)
    if not rec:
        return curie, None
    ident = rec.get("id", {}).get("identifier", curie)
    label = rec.get("id", {}).get("label")
    return ident, label

def name_resolve_disease_mondo(text: str, debug: dict):
    q = norm_text(text)
    params = {"string": q, "limit": 10, "only_prefixes": "MONDO"}
    r = requests.get(NAME_RESOLVER, params=params, timeout=30)
    r.raise_for_status()
    cands = r.json()
    debug["attempts"].append({"mode": "disease_mondo", "query": q, "candidates": cands})
    if not cands:
        return {"curie": None, "label": None, "notes": "NO_MONDO_MATCH", "debug": debug}
    best = cands[0]
    curie2, label = canonicalize(best["curie"])
    return {"curie": curie2, "label": label or best.get("label"), "notes": "DISEASE_MONDO", "debug": debug}

def hgnc_symbol_to_id(symbol: str, debug: dict):
    """
    Returns HGNC numeric id as string (e.g. "11998") or None.
    ClinicalTables response shape:
      [count, [idlist], null, [[display_fields...]]]
    """
    sym = norm_gene_symbol(symbol)
    params = {"sf": "symbol", "terms": sym}
    r = requests.get(HGNC_CT_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    debug["attempts"].append({"mode": "hgnc_clinicaltables", "query": sym, "response": data})

    if not isinstance(data, list) or len(data) < 2:
        return None
    idlist = data[1]
    if not idlist:
        return None
    return str(idlist[0])

def resolve_gene(symbol: str, debug: dict):
    sym = norm_gene_symbol(symbol)
    hgnc_id = hgnc_symbol_to_id(sym, debug)
    if not hgnc_id:
        return {"curie": None, "label": None, "notes": "NO_HGNC_MATCH", "debug": debug}

    curie = f"HGNC:{hgnc_id}"
    curie2, label = canonicalize(curie)
    # If Node Normalizer doesn't return a label, fall back to the symbol
    return {"curie": curie2, "label": label or sym, "notes": "GENE_HGNC", "debug": debug}

def resolve_variant_as_gene_proxy(text: str, debug: dict):
    """
    Your questions use underspecified variant phrases (e.g. 'BRAF mutation').
    For ROBOKOP node-pairs evaluation, treat these as GENE proxies:
      'BRAF mutation' -> BRAF
      'p53 mutation'  -> TP53
    """
    q = norm_text(text)
    q = re.sub(r"\bmutations\b", "mutation", q, flags=re.IGNORECASE)
    first = norm_gene_symbol(q.split()[0])

    notes = ["VARIANT_PROXY_TO_GENE"]
    if first == "P53":
        first = "TP53"
        notes.append("P53_TO_TP53")

    res = resolve_gene(first, debug)
    if res["curie"]:
        res["notes"] = ";".join(notes + [res["notes"]])
        return res

    return {"curie": None, "label": None, "notes": "NO_VARIANT_PROXY_MATCH", "debug": debug}

def resolve_one(text: str):
    debug = {"query": text, "attempts": []}
    t = norm_text(text)

    # Variant phrases in your dataset should be treated as gene proxies for ROBOKOP nodes.
    if is_mutation_phrase(t):
        return resolve_variant_as_gene_proxy(t, debug)

    if is_gene_like(t):
        return resolve_gene(t, debug)

    return name_resolve_disease_mondo(t, debug)

def main(node_pairs_tsv: str, out_tsv: str, debug_jsonl: str):
    with open(node_pairs_tsv, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        rows = list(rdr)

    cache = {}
    debug_lines = []

    def resolve_cached(text):
        if text not in cache:
            res = resolve_one(text)
            cache[text] = res
            debug_lines.append(res["debug"])
            time.sleep(0.05)
        return cache[text]

    out_rows = []
    for r in rows:
        qid = r["question_id"]
        subj = r["subject"]
        obj = r["object"]

        subj_res = resolve_cached(subj)
        obj_res = resolve_cached(obj)

        out_rows.append({
            "question_id": qid,
            "subject_text": subj,
            "subject_curie": subj_res["curie"] or "",
            "subject_label": subj_res["label"] or "",
            "object_text": obj,
            "object_curie": obj_res["curie"] or "",
            "object_label": obj_res["label"] or "",
            "notes": f"SUBJ:{subj_res['notes']}|OBJ:{obj_res['notes']}"
        })

    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "subject_text","subject_curie","subject_label",
                "object_text","object_curie","object_label",
                "notes"
            ],
            delimiter="\t"
        )
        w.writeheader()
        w.writerows(out_rows)

    with open(debug_jsonl, "w", encoding="utf-8") as f:
        for d in debug_lines:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_tsv}")
    print(f"Wrote: {debug_jsonl}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python step2_resolve.py node_pairs.tsv resolved_node_pairs.tsv resolution_debug.jsonl")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])