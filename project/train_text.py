import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          Trainer,
                          TrainingArguments,
                          DataCollatorWithPadding)


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
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


if __name__ == '__main__':
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    load_dotenv()

    HF_TOKEN = os.getenv("HF_TOKEN")
    PRETRAINED_PATH = Path("models/category_text")
    DATASET_PATH = Path("../data/texts.parquet")
    MODEL_NAME = "DeepPavlov/rubert-base-cased"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, token=HF_TOKEN)

    dataset = AirsoftTextDataset(pq_path=DATASET_PATH, tokenizer=tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir="./rubert_model",
        per_device_train_batch_size=64,
        fp16=True,
        learning_rate=2e-5,
        num_train_epochs=5,
        logging_steps=200,
        save_strategy="steps",
        save_steps=1000,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(PRETRAINED_PATH)
    tokenizer.save_pretrained(PRETRAINED_PATH)
