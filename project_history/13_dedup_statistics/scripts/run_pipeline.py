import argparse
from pathlib import Path
from analyze import analyze

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True, help="KG TSV (deduplicated), e.g. data/robokop_triples_deduplicated.tsv")
    ap.add_argument("--gpt", required=True, help="GPT JSONL, e.g. data/gpt_runs_v1.json")
    ap.add_argument("--out_dir", required=True, help="Output directory, e.g. out/v1")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = analyze(args.kg, args.gpt)

    # No console output; write to files only
    res["summary"].to_csv(out_dir / "summary.tsv", sep="\t", index=False)
    res["per_question"].to_csv(out_dir / "per_question_metrics.tsv", sep="\t", index=False)
    res["per_run"].to_csv(out_dir / "per_run_metrics.tsv", sep="\t", index=False)
    res["top15_questions"].to_csv(out_dir / "top15_questions.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()
