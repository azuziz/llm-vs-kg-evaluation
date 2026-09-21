#!/usr/bin/env python3
import pandas as pd
import json
from itertools import combinations

def edgeset(s):
    try:
        j=json.loads(s)
        return {(e["subject"],e["predicate"],e["object"])
                for e in j.get("edges",[])}
    except:
        return set()

def mean_pairwise_jaccard(sets):
    if len(sets)<2:
        return 1.0
    vals=[]
    for a,b in combinations(sets,2):
        if not a and not b:
            vals.append(1)
        else:
            vals.append(len(a&b)/len(a|b))
    return sum(vals)/len(vals)

for pv in ["v2","v3","v4"]:
    df=pd.read_csv(f"~/gpt/12_stats/{pv}/gpt_runs.csv")
    df=df[df.status=="ok"]

    scores=[]
    for q,sub in df.groupby("question_id"):
        sets=[edgeset(x) for x in sub.edges_json]
        scores.append(mean_pairwise_jaccard(sets))

    print(pv,"mean semantic stability:",sum(scores)/len(scores))
