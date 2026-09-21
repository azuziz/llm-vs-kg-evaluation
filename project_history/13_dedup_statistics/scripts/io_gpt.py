import json
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

Triple = Tuple[str, str, str]

@dataclass(frozen=True)
class GptRun:
    question_id: str
    run_id: int
    curie1: str
    curie2: str
    edges: Set[Triple]
    n_edges: int

def _edges_from_edges_json(cell: str) -> Set[Triple]:
    """
    edges_json format example:
      {"edges":[{"predicate":"biolink:...","subject":"MONDO:...","object":"NCBIGene:..."}]}
    Returns a set of (subject, predicate, object).
    """
    if cell is None:
        return set()
    s = str(cell).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return set()

    try:
        obj = json.loads(s)
    except Exception:
        return set()

    edges = obj.get("edges", [])
    out: Set[Triple] = set()
    if isinstance(edges, list):
        for e in edges:
            if not isinstance(e, dict):
                continue
            subj = e.get("subject", "")
            pred = e.get("predicate", "")
            objj = e.get("object", "")
            if isinstance(subj, str) and isinstance(pred, str) and isinstance(objj, str):
                subj = subj.strip()
                pred = pred.strip()
                objj = objj.strip()
                if subj and pred and objj:
                    out.add((subj, pred, objj))
    return out

def load_gpt_runs_jsonl(path_jsonl: str) -> List[GptRun]:
    """
    v1 JSONL schema:
      question_id, run_id, curie1, curie2, edge_keys (list of [s,p,o]), n_edges
    """
    runs: List[GptRun] = []
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("question_id", "")).strip()
            run_id = int(obj.get("run_id", -1))
            curie1 = str(obj.get("curie1", "")).strip()
            curie2 = str(obj.get("curie2", "")).strip()

            edge_keys = obj.get("edge_keys", [])
            edges: Set[Triple] = set()
            if isinstance(edge_keys, list):
                for it in edge_keys:
                    if isinstance(it, list) and len(it) == 3:
                        s, p, o = it
                        if isinstance(s, str) and isinstance(p, str) and isinstance(o, str):
                            edges.add((s.strip(), p.strip(), o.strip()))

            n_edges = int(obj.get("n_edges", len(edges)))
            runs.append(GptRun(qid, run_id, curie1, curie2, edges, n_edges))
    return runs

def load_gpt_runs_v2v4_csv(path_csv: str) -> List[GptRun]:
    """
    v2-v4 CSV schema (your header):
      prompt_version,question_id,run_id,subject_curie,object_curie,edges_json,status,error

    edges_json contains {"edges":[{"predicate","subject","object"}, ...]}
    """
    df = pd.read_csv(path_csv, dtype=str).fillna("")
    needed = {"question_id", "run_id", "subject_curie", "object_curie", "edges_json"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"GPT v2-v4 CSV missing columns: {sorted(missing)}")

    runs: List[GptRun] = []
    for _, row in df.iterrows():
        qid = str(row["question_id"]).strip()
        run_id = int(str(row["run_id"]).strip() or -1)
        curie1 = str(row["subject_curie"]).strip()
        curie2 = str(row["object_curie"]).strip()
        edges = _edges_from_edges_json(row["edges_json"])
        runs.append(GptRun(qid, run_id, curie1, curie2, edges, len(edges)))

    return runs

def load_gpt_runs(path: str) -> List[GptRun]:
    """
    Dispatcher:
      - .json/.jsonl -> v1 loader
      - .csv -> if it has edges_json + subject_curie/object_curie -> v2-v4 loader
    """
    p = path.lower()
    if p.endswith(".json") or p.endswith(".jsonl"):
        return load_gpt_runs_jsonl(path)

    if p.endswith(".csv"):
        # Peek header to decide
        df0 = pd.read_csv(path, nrows=1, dtype=str)
        cols = set(df0.columns)
        if {"edges_json", "subject_curie", "object_curie", "question_id", "run_id"}.issubset(cols):
            return load_gpt_runs_v2v4_csv(path)
        raise ValueError(f"Unrecognized GPT CSV schema. Columns: {sorted(cols)}")

    raise ValueError(f"Unsupported GPT runs file type: {path}")

def group_runs_by_question(runs: List[GptRun]) -> Dict[str, List[GptRun]]:
    out: Dict[str, List[GptRun]] = {}
    for r in runs:
        out.setdefault(r.question_id, []).append(r)
    for qid in out:
        out[qid] = sorted(out[qid], key=lambda x: x.run_id)
    return out
