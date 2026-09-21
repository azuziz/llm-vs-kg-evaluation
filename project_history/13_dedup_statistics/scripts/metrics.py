from dataclasses import dataclass
from typing import Set, Tuple

Triple = Tuple[str, str, str]

@dataclass
class RunMetrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    gpt_edges: int
    kg_edges: int
    matched_edges: int

def f1_from_pr(p: float, r: float) -> float:
    if p <= 0.0 and r <= 0.0:
        return 0.0
    denom = (p + r)
    return (2.0 * p * r / denom) if denom > 0 else 0.0

def compute_run_metrics(gpt: Set[Triple], kg: Set[Triple]) -> RunMetrics:
    inter = gpt.intersection(kg)
    tp = len(inter)
    fp = len(gpt) - tp
    fn = len(kg) - tp

    precision = (tp / len(gpt)) if len(gpt) > 0 else 0.0
    recall = (tp / len(kg)) if len(kg) > 0 else 0.0
    f1 = f1_from_pr(precision, recall)

    return RunMetrics(
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        gpt_edges=len(gpt), kg_edges=len(kg), matched_edges=tp
    )
