import csv, time, random
from openai import OpenAI

client = OpenAI()

questions = [
    "is there any association of LCH with BRAF mutation?",
    "is there any bias of sexes in LCH",
    "is LCH associated with neurodegeneration?",
    "What is the molecular mechanism of sorafenib?",
    "What regulates the expression of cyclin-dependent kinases?",
]

N_RUNS = 10                                                                                         #may vary
MASTER_FILE = "data/responses/all_responses.csv"

# Create master CSV header
with open(MASTER_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "question", "run", "response"])

for q_id, question in enumerate(questions):
    q_file = f"data/responses/Q{q_id+1:02d}.csv"
    print(f"\n--- Question {q_id+1}/{len(questions)} ---")

    with open(q_file, "w", encoding="utf-8", newline="") as qf:
        qwriter = csv.writer(qf)
        qwriter.writerow(["question_id", "question", "run", "response"])

        for run in range(1, N_RUNS + 1):
            print(f"Run {run}/{N_RUNS}...", end=" ")
            while True:
                try:
                    completion = client.chat.completions.create(
                        model="gpt-5-nano",
                        messages=[{"role": "user", "content": question}],
                        temperature=1.0
                    )
                    response = completion.choices[0].message.content.strip()
                    break
                except Exception:
                    print("Rate-limit; waiting…")
                    time.sleep(5)

            row = [q_id, question, run, response]
            qwriter.writerow(row)
            with open(MASTER_FILE, "a", encoding="utf-8", newline="") as mf:
                csv.writer(mf).writerow(row)

            time.sleep(random.uniform(0.2, 0.4))
    print("✓ done")
print("All responses saved in data/responses/")
