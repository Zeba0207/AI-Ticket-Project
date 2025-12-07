import json

input_file = "../data/annotated/seed_for_labeling.jsonl"
output_file = "../data/annotated/seed_for_labeling.json"

data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Converted JSONL → JSON successfully!")
print(f"Saved to: {output_file}")
