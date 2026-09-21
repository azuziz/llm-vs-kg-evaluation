import json, numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import glob, csv, os

client = OpenAI()

def embed(text):
    emb = client.embeddings.create(model="text-embedding-3-large", input=text)
    return emb.data[0].embedding

def compare_embeddings():
    results = []
    for qfile in glob.glob("data/extracted_triples/*.json"):
        robokop_file = "data/robokop_triples/BRAF.json"  # placeholder
        kg = json.load(open(robokop_file))
        kg_texts = [" ".join(k) for k in kg]
        kg_embs = [embed(t) for t in kg_texts]

        data = json.load(open(qfile))
        for run in data:
            gpt_text = " ".join([f"{t}" for t in run["triples"]])
            gpt_emb = embed(gpt_text)
            sims = cosine_similarity([gpt_emb], kg_embs)[0]
            results.append({
                "question": run["question"],
                "run": run["run"],
                "mean_similarity": float(np.mean(sims)),
                "max_similarity": float(np.max(sims))
            })

    os.makedirs("results", exist_ok=True)
    with open("results/similarity_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

if __name__ == "__main__":
    compare_embeddings()
