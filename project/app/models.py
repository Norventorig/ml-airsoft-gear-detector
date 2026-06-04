import os
from io import BytesIO
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import timm
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models, transforms

from .settings import TEXT_MODEL_DIR, CATEGORY_IMAGE_MODEL, SUBCATEGORY_IMAGE_MODEL

ID2CATEGORY = {
    0: 'Страйкбольное оружие',
    1: 'Аксессуары и Запчасти',
    2: 'Снаряжение и защита'
}

ID2SUBCATEGORY = {
    0: 'ak', 1: 'backpack', 2: 'helmet', 3: 'HK', 4: 'M serias', 5: 'machinegun',
    6: 'pistol', 7: 'pouch', 8: 'rifle', 9: 'shutgun', 10: 'vest'
}


class ModelPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self._load_models()

    def _load_models(self):
        """Инициализация архитектур и безопасная загрузка весов."""

        if os.path.exists(TEXT_MODEL_DIR):
            self.text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
            self.text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR)
            self.text_model.to(self.device).eval()
        else:
            self.text_model = None

        self.cat_img_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(ID2CATEGORY))
        if os.path.exists(CATEGORY_IMAGE_MODEL):
            self.cat_img_model.load_state_dict(
                torch.load(CATEGORY_IMAGE_MODEL, map_location=self.device, weights_only=True)
            )
        self.cat_img_model.to(self.device).eval()

        self.subcat_img_model = models.resnet18()
        num_ftrs = self.subcat_img_model.fc.in_features
        self.subcat_img_model.fc = nn.Linear(num_ftrs, len(ID2SUBCATEGORY))

        if os.path.exists(SUBCATEGORY_IMAGE_MODEL):
            self.subcat_img_model.load_state_dict(
                torch.load(SUBCATEGORY_IMAGE_MODEL, map_location=self.device, weights_only=True)
            )
        self.subcat_img_model.to(self.device).eval()

    def _download_and_preprocess_image(self, url: str) -> torch.Tensor:
        """Загрузка изображения по HTTP-ссылке и приведение к тензору."""
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')
            return self.img_transform(img).unsqueeze(0).to(self.device)
        except Exception:
            return torch.zeros(1, 3, 224, 224).to(self.device)

    @torch.no_grad()
    def predict(self, text: str, photos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        predictions = []

        text_probs = torch.zeros(len(ID2CATEGORY)).to(self.device)
        if self.text_model:
            inputs = self.text_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_outputs = self.text_model(**inputs)
            text_probs = F.softmax(text_outputs.logits, dim=-1).squeeze(0)

        for idx, photo in enumerate(photos, start=1):
            img_tensor = self._download_and_preprocess_image(photo["url"])

            cat_img_outputs = self.cat_img_model(img_tensor)
            cat_img_probs = F.softmax(cat_img_outputs, dim=-1).squeeze(0)

            combined_probs = (text_probs + cat_img_probs) / 2.0
            cat_id = torch.argmax(combined_probs).item()

            category_name = ID2CATEGORY.get(cat_id, "Неизвестная категория")
            category_confidence = combined_probs[cat_id].item()

            subcat_outputs = self.subcat_img_model(img_tensor)
            subcat_probs = F.softmax(subcat_outputs, dim=-1).squeeze(0)
            subcat_id = torch.argmax(subcat_probs).item()

            subcategory_name = ID2SUBCATEGORY.get(subcat_id, "Неизвестная подкатегория")

            predictions.append({
                "object_id": str(idx),
                "category": category_name,
                "subcategory": subcategory_name,
                "confidence": round(category_confidence, 2),
                "photo_ids": [photo["photo_id"]]
            })

        return predictions


ml_pipeline = ModelPipeline()
