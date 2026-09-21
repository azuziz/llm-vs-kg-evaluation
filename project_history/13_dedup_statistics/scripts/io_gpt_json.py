import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

Triple = Tuple[str, str, str]

@dataclass(frozen=True)
class GptRun:
    question_id: str
    run_id: int
    curie1: str
    curie2: str
    edges: Set[Triple]   # parsed from edge_keys
    n_edges: int

def _parse_edge_keys(edge_keys) -> Set[Triple]:
    """
    edge_keys is expected to be a list of [subject, predicate, object].
    """
    out: Set[Triple] = set()
    if not edge_keys:
        return out
    for item in edge_keys:
        if not isinstance(item, list) or len(item) != 3:
            continue
        s, p, o = item
        if not (isinstance(s, str) and isinstance(p, str) and isinstance(o, str)):
            continue
        out.add((s, p, o))
    return out

def load_gpt_runs_jsonl(path_jsonl: str) -> List[GptRun]:
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
            edges = _parse_edge_keys(edge_keys)
            n_edges = int(obj.get("n_edges", len(edges)))
            runs.append(GptRun(qid, run_id, curie1, curie2, edges, n_edges))
    return runs

def group_runs_by_question(runs: List[GptRun]) -> Dict[str, List[GptRun]]:
    out: Dict[str, List[GptRun]] = {}
    for r in runs:
        out.setdefault(r.question_id, []).append(r)
    # stable ordering by run_id
    for qid in out:
        out[qid] = sorted(out[qid], key=lambda x: x.run_id)
    return out
