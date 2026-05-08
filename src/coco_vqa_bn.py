import os
import json
import random
from glob import glob

INPUT_FOLDER = r"D:\NLP Materials\BanglaVision\output\translated"

OUTPUT_FILE = r"D:\NLP Materials\BanglaVision\output\vqa_bn_dataset.json"


QUESTIONS = [

    "এই ছবিতে কী দেখা যাচ্ছে?",

    "ছবিটি বর্ণনা করুন।",

    "ছবিতে কী আছে?",

    "এই দৃশ্যে কী দেখা যাচ্ছে?",

    "ছবির মধ্যে কী রয়েছে?",

    "ছবিটি সম্পর্কে বলুন।"
]

all_files = glob(os.path.join(INPUT_FOLDER, "*.json"))

print("Total batch files:", len(all_files))

final_data = []

for file_path in all_files:

    print("Processing:", os.path.basename(file_path))

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:

        caption_bn = item.get("caption_bn", "").strip()

        if caption_bn == "__FAILED__":
            continue

        if len(caption_bn) < 3:
            continue

        vqa_item = {

            "image": item["image"],

            "question_bn": random.choice(QUESTIONS),

            "answer_bn": caption_bn
        }

        final_data.append(vqa_item)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    json.dump(
        final_data,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\nDone.")
print("Total VQA samples:", len(final_data))
print("Saved to:", OUTPUT_FILE)