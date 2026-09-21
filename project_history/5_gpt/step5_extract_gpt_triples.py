import csv
import json
import os
import sys
from glob import glob

def extract_output_text(resp_json: dict) -> str:
    return resp_json["output"][0]["content"][0]["text"]

def main(gpt_raw_dir: str, out_tsv: str):
    files = sorted(glob(os.path.join(gpt_raw_dir, "Q*.json")))
    rows = []
    for fp in files:
        qid = os.path.splitext(os.path.basename(fp))[0]
        with open(fp, "r", encoding="utf-8") as f:
            resp = json.load(f)
        parsed = json.loads(extract_output_text(resp))
        triples = parsed.get("triples") or []
        for t in triples:
            rows.append({
                "question_id": qid,
                "subject": t.get("subject",""),
                "predicate": t.get("predicate",""),
                "object": t.get("object","")
            })

    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question_id","subject","predicate","object"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote: {out_tsv} (rows={len(rows)})")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python step5_extract_gpt_triples.py gpt_raw gpt_triples.tsv")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
