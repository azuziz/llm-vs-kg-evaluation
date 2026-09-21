import csv, json, os, time
from openai import OpenAI

client = OpenAI()
# TEMP_EXTRACT = 0.0 no need since it dont work for gpt-5
# TOPP_EXTRACT = 1.0  no need since it dont work for gpt-5

def extract_triples(text):
    prompt = f"Extract biomedical triples (subject, relation, object) from this text:\n\n{text}\n\nReturn JSON list of triples."
    while True:
        try:
            res = client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": prompt}]
            )
            content = res.choices[0].message.content
            # Defensive: sometimes model returns plain text
            return json.loads(content) if content.strip().startswith("[") else []
        except json.JSONDecodeError:
            print("Warning: non-JSON output, retrying...")
            time.sleep(3)
        except Exception as e:
            print("Retrying extraction...", e)
            time.sleep(5)


def process_all():
    os.makedirs("data/extracted_triples", exist_ok=True)
    for file in os.listdir("data/responses"):
        if not file.endswith(".csv"): continue
        out_file = "data/extracted_triples/" + file.replace(".csv", ".json")
        triples_all = []
        with open("data/responses/" + file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                triples = extract_triples(row["response"])
                triples_all.append({
                    "question": row["question"],
                    "run": row["run"],
                    "triples": triples
                })
        with open(out_file, "w") as f:
            json.dump(triples_all, f, indent=2)

if __name__ == "__main__":
    process_all()
