import json

file_path = r"D:\NLP Materials\BanglaVision\output\translated\coco_bn_final.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

clean_data = [item for item in data if item["caption_bn"] != "__FAILED__"]

print("Original:", len(data))
print("Clean:", len(clean_data))

# Save clean version
with open(r"D:\NLP Materials\BanglaVision\output\translated\coco_bn_clean.json", "w", encoding="utf-8") as f:
    json.dump(clean_data, f, indent=2, ensure_ascii=False)

print("Clean dataset saved")