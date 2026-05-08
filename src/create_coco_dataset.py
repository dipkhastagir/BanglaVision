import os

def create_dataset(data, image_folder):
    dataset = []

    for item in data["annotations"]:
        image_id = item["image_id"]
        caption = item["caption"]

        image_path = os.path.join(image_folder, f"{image_id:012d}.jpg")

        dataset.append({
            "image": image_path,
            "caption": caption
        })

    return dataset