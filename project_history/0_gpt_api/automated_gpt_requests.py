import csv
import time
from openai import OpenAI
import random

client = OpenAI()

questions = [
    "is there any association of LCH with BRAF mutation?"
]

N_RUNS = 20  # number of repetitions per question
OUTPUT_FILE = "gpt_responses.csv"

# Create CSV header
with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "question", "run", "response"])

for q_id, question in enumerate(questions):
    print(f"\n--- Running question {q_id + 1}/{len(questions)} ---")

    for run in range(1, N_RUNS + 1):
        print(f"Request {run}/20...", end=" ")      # change 20 runs to new run number

        while True:
            try:
                completion = client.chat.completions.create(
                    model="gpt-5-nano-2025-08-07",  # ✅ Call GPT-5 nano snapshot 2025.08.07
                    messages=[{"role": "user", "content": question}],
                    temperature=1.0  # Keep randomness for statistical analysis
                )
                response_text = completion.choices[0].message.content.strip()
                break

            except Exception as e:
                print("Rate limit, waiting 5 seconds...")
                time.sleep(5)

        # Save response
        with open(OUTPUT_FILE, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([q_id, question, run, response_text])

        # small random delay to avoid rate-limit bursts
        time.sleep(random.uniform(0.1, 0.3))

    print("✓ Done")

print("\nAll requests completed. Responses saved in gpt_responses.csv")
