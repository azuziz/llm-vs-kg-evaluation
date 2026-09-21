import pandas as pd
from typing import Dict, Set, Tuple

Triple = Tuple[str, str, str]

def load_kg_triples_by_question(path_tsv: str) -> Dict[str, Set[Triple]]:
    """
    Load deduplicated ROBOKOP triples and return:
      question_id -> set of (subject, predicate, object)
    """
    df = pd.read_csv(path_tsv, sep="\t", dtype=str).fillna("")
    needed = {"question_id", "subject", "predicate", "object"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"KG TSV missing columns: {sorted(missing)}")

    out: Dict[str, Set[Triple]] = {}
    for qid, subdf in df.groupby("question_id", dropna=False):
        triples = set(zip(subdf["subject"], subdf["predicate"], subdf["object"]))
        out[str(qid)] = triples
    return out
