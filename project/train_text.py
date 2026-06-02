from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          Trainer,
                          TrainingArguments,
                          DataCollatorWithPadding)

import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from dotenv import load_dotenv
import torch
import os

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
PRETRAINED_PATH = Path("models/category_text")
DATASET_PATH = Path("../data/texts.parquet")
MODEL_NAME = "DeepPavlov/rubert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, token=HF_TOKEN)
model.to(device)


class AirsoftTextDataset(Dataset):
    def __init__(self, pq_path, tokenizer):
        self.df = pd.read_parquet(pq_path)
        self.labels = self.df["id"].tolist()

        texts = self.df["text"].tolist()
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=256
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

dataset = AirsoftTextDataset(pq_path=DATASET_PATH, tokenizer=tokenizer)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
args = TrainingArguments(output_dir="./rubert_model",
                         per_device_train_batch_size=32,
                         fp16=True,
                         learning_rate=2e-5,
                         num_train_epochs=5,
                         logging_steps=200,
                         save_strategy="steps",
                         save_steps=1000,
                         report_to="none")
trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=data_collator)
trainer.train()
model.save_pretrained(PRETRAINED_PATH)
tokenizer.save_pretrained(PRETRAINED_PATH)
