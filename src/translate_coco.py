import os
import json
import time
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor


COCO_JSON = r"D:\NLP Materials\BanglaVision\data\coco\annotations_trainval2017\annotations\captions_train2017.json"
IMAGE_FOLDER = r"D:\NLP Materials\BanglaVision\data\coco\train2017"
OUTPUT_FOLDER = r"D:\NLP Materials\BanglaVision\output\translated"

BATCH_SIZE = 1000
NUM_WORKERS = 5  



os.makedirs(OUTPUT_FOLDER, exist_ok=True)

with open(COCO_JSON, "r") as f:
    data = json.load(f)

annotations = data["annotations"]

print("Total items:", len(annotations))



def safe_translate(text, retries=2):
    for i in range(retries):
        try:
            result = GoogleTranslator(source='en', target='bn').translate(text)

            if result and result.strip().lower() != text.strip().lower():
                return result

        except Exception:
            pass

        time.sleep(1)  

    return "__FAILED__"



def process_chunk(chunk):
    results = []

    for item in chunk:
        image_id = item["image_id"]
        caption = item["caption"]

        image_path = os.path.join(IMAGE_FOLDER, f"{image_id:012d}.jpg")

        translated = safe_translate(caption)

        results.append({
            "image": image_path,
            "caption_en": caption,
            "caption_bn": translated
        })

    return results


for start in range(0, len(annotations), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(annotations))
    output_file = os.path.join(OUTPUT_FOLDER, f"batch_{start}_{end}.json")

    if os.path.exists(output_file):
        print(f"Skipping {start} to {end}")
        continue

    print(f"Processing {start} to {end}")

    batch_data = annotations[start:end]

    chunk_size = len(batch_data) // NUM_WORKERS + 1
    chunks = [batch_data[i:i + chunk_size] for i in range(0, len(batch_data), chunk_size)]

    results = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for res in executor.map(process_chunk, chunks):
            results.extend(res)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved batch {start} to {end}")

    time.sleep(1)  


print("Translation process completed.")