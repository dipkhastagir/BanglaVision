import os
import json
import time
from deep_translator import GoogleTranslator

OUTPUT_FOLDER = r"D:\NLP Materials\BanglaVision\output\translated"

RETRIES = 3
SLEEP_TIME = 1


def retry_translate(text, retries=RETRIES):
    for _ in range(retries):
        try:
            result = GoogleTranslator(source='en', target='bn').translate(text)

            if result and result.strip().lower() != text.strip().lower():
                return result

        except Exception:
            pass

        time.sleep(SLEEP_TIME)

    return "__FAILED__"


def process_file(file_path):
    print(f"\nProcessing: {os.path.basename(file_path)}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = False
    fail_count = 0
    fixed_count = 0

    for item in data:
        if item["caption_bn"] == "__FAILED__":
            fail_count += 1

            new_translation = retry_translate(item["caption_en"])

            if new_translation != "__FAILED__":
                item["caption_bn"] = new_translation
                updated = True
                fixed_count += 1

    print(f"Total failed found: {fail_count}")
    print(f"Successfully fixed: {fixed_count}")

    if updated:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print("File updated")
    else:
        print("No changes needed")


def main():
    files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".json")]

    print(f"Total batch files: {len(files)}")

    for file in files:
        file_path = os.path.join(OUTPUT_FOLDER, file)
        process_file(file_path)

    print("\nRetry process completed!")


if __name__ == "__main__":
    main()