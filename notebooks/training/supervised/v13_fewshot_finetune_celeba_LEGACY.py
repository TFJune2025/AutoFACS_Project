#!/usr/bin/env python
# Few-Shot Fine-Tuning Script for CelebA Subset using V13

import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForImageClassification, AutoImageProcessor, Trainer, TrainingArguments
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
CELEBA_SUBSET_DIR = "/Users/natalyagrokh/AI/ml_expressions/img_datasets/celeba_subset_annotated"
MODEL_PATH = "/Users/natalyagrokh/AI/ml_expressions/img_expressions/V13"
OUTPUT_DIR = "./v13_celeba_finetuned"
LABELS = sorted(os.listdir(CELEBA_SUBSET_DIR))
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -----------------------------
# Load Model and Processor
# -----------------------------
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH).to(DEVICE)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

# Freeze all layers except classifier
for name, param in model.named_parameters():
    param.requires_grad = "classifier" in name or "head" in name

# -----------------------------
# Load Dataset
# -----------------------------
def img_loader(example):
    image = Image.open(example["file"]).convert("RGB")
    return processor(image, return_tensors="pt")

def collate_fn(batch):
    inputs = processor([x["image"] for x in batch], return_tensors="pt", padding=True)
    inputs["labels"] = torch.tensor([x["label"] for x in batch])
    return inputs

def prepare_dataset():
    data = []
    for idx, label in enumerate(LABELS):
        label_dir = os.path.join(CELEBA_SUBSET_DIR, label)
        for fname in os.listdir(label_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                data.append({
                    "file": os.path.join(label_dir, fname),
                    "label": idx,
                    "image": Image.open(os.path.join(label_dir, fname)).convert("RGB")
                })
    return DatasetDict({
        "train": load_dataset("imagefolder", data_dir=CELEBA_SUBSET_DIR, split="train[:90%]"),
        "eval": load_dataset("imagefolder", data_dir=CELEBA_SUBSET_DIR, split="train[90%:]")
    })

ds = prepare_dataset()
ds = ds.cast_column("image", processor.image_processor)

# -----------------------------
# Evaluation Metrics
# -----------------------------
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average="weighted")
    }

# -----------------------------
# Training
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=3,
    learning_rate=1e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["eval"],
    tokenizer=processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
