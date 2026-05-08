import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model


DATA_PATH = "/content/drive/MyDrive/BanglaVision/coco_bn_clean.json"
IMAGE_ROOT = "/content/drive/MyDrive/BanglaVision/train2017"
OUTPUT_DIR = "/content/drive/MyDrive/BanglaVision/models/blip2_bn"

BATCH_SIZE = 4
EPOCHS = 1
LR = 2e-5

os.makedirs(OUTPUT_DIR, exist_ok=True)


class CocoDataset(Dataset):
    def __init__(self, json_path, image_root, processor):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = item["image"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.image_root, os.path.basename(image_path))

        image = Image.open(image_path).convert("RGB")
        caption = item["caption_bn"]

        inputs = self.processor(images=image, text=caption, return_tensors="pt")

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

print("Loading model...")

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    device_map="auto"
)


print("Applying LoRA...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)


dataset = CocoDataset(DATA_PATH, IMAGE_ROOT, processor)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

model.train()

print("Starting training...")

for epoch in range(EPOCHS):

    total_loss = 0

    for step, batch in enumerate(dataloader):

        batch = {k: v.to(model.device) for k, v in batch.items()}

        outputs = model(**batch, labels=batch["input_ids"])

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item()}")

        if step % 2000 == 0 and step != 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_{epoch}_{step}")
            model.save_pretrained(ckpt_path)
            print(f"Checkpoint saved at {ckpt_path}")

    print(f"Epoch {epoch} finished | Avg Loss: {total_loss / len(dataloader)}")


print("Saving final model...")

model.save_pretrained(OUTPUT_DIR)

print("Training complete!")