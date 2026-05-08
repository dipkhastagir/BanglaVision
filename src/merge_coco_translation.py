import os
import json

OUTPUT_FOLDER = r"D:\NLP Materials\BanglaVision\output\translated"
FINAL_FILE = os.path.join(OUTPUT_FOLDER, "coco_bn_final.json")

all_data = []

files = sorted([f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".json")])

for file in files:
    file_path = os.path.join(OUTPUT_FOLDER, file)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        all_data.extend(data)

print("Total merged samples:", len(all_data))

with open(FINAL_FILE, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print("Final dataset ready:", FINAL_FILE)