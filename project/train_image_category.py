# import io
# import os
# import requests
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm
#
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# import timm
#
# # Глобальные параметры
# NUM_CLASSES = 3
# BATCH_SIZE = 32  # Оптимально для RTX 3060 (12GB)
# EPOCHS = 5
# LEARNING_RATE = 1e-4
# MODEL_SAVE_PATH = "best_image_classifier.pth"
#
# # Трансформации для картинок
# train_transforms = T.Compose([
#     T.Resize((224, 224)),
#     T.RandomHorizontalFlip(),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
#
# class ParquetImageDataset(Dataset):
#     def __init__(self, parquet_path, transform=None):
#         self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
#         self.transform = transform
#
#         if self.df['label'].dtype == 'object':
#             self.labels = self.df['label'].astype('category').cat.codes.values
#         else:
#             self.labels = self.df['label'].values
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         url = self.df.iloc[idx]['url']
#         label = self.labels[idx]
#
#         try:
#             response = requests.get(url, timeout=5)
#             image = Image.open(io.BytesIO(response.content)).convert('RGB')
#         except Exception:
#             image = Image.new('RGB', (224, 224), color=0)
#
#         if self.transform:
#             image = self.transform(image)
#
#         return image, torch.tensor(label, dtype=torch.long)
#
#
# # ГЛАВНАЯ КОРРЕКТИРОВКА: Весь код запуска должен быть внутри этой функции
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Используется устройство: {device}")
#
#     # Инициализация датасета и загрузчика
#     dataset = ParquetImageDataset(r"..\data\images.parquet", transform=train_transforms)
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
#
#     # Инициализация модели
#     model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
#     model = model.to(device)
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#
#     # Новый синтаксис вместо torch.cuda.amp.GradScaler()
#     scaler = torch.amp.GradScaler('cuda')
#
#     print("Начало обучения...")
#     best_loss = float('inf')
#
#     for epoch in range(EPOCHS):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#
#         loop = tqdm(dataloader, desc=f"Эпоха {epoch + 1}/{EPOCHS}")
#         for images, labels in loop:
#             images = images.to(device, non_blocking=True)
#             labels = labels.to(device, non_blocking=True)
#
#             optimizer.zero_grad()
#
#             # Новый синтаксис вместо torch.cuda.amp.autocast()
#             with torch.amp.autocast('cuda'):
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#
#             running_loss += loss.item() * images.size(0)
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
#
#             loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)
#
#         epoch_loss = running_loss / len(dataset)
#         print(f"Итог эпохи {epoch + 1}: Loss: {epoch_loss:.4f} | Точность: {100.0 * correct / total:.2f}%")
#
#         if epoch_loss < best_loss:
#             best_loss = epoch_loss
#             torch.save(model.state_dict(), MODEL_SAVE_PATH)
#             print(f"--> Модель сохранена в файл: {MODEL_SAVE_PATH}")
#
#     print("Обучение завершено успешно!")
#
#
# # КРИТИЧЕСКИ ВАЖНО ДЛЯ WINDOWS
# if __name__ == '__main__':
#     main()
import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

# 1. ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ (Оптимизировано под RTX 3060 12GB)
NUM_CLASSES = 3
BATCH_SIZE = 128  # Увеличено с 32 для полной загрузки GPU
EPOCHS = 5
LEARNING_RATE = 3e-4  # Чуть выше для увеличенного батча
MODEL_SAVE_PATH = r"models\category_image\best_image_classifier.pth"
CACHED_IMG_DIR = r"..\data\images_cached"

# 2. БЫСТРЫЕ ТРАНСФОРМАЦИИ (Albumentations на базе OpenCV быстрее PIL)
train_transforms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class LocalImageDataset(Dataset):
    def __init__(self, parquet_path, img_dir, transform=None):
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        if self.df['label'].dtype == 'object':
            self.labels = self.df['label'].astype('category').cat.codes.values
        else:
            self.labels = self.df['label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = os.path.join(self.img_dir, f"{idx}.jpg")

        # Быстрое чтение локального файла с диска
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception:
            # Черный квадрат, если файл поврежден или отсутствует
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)


def main():
    # Проверка доступности GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    if device.type == 'cuda':
        # Включаем оптимизацию под архитектуру Ampere (RTX 3060)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Включены оптимизации TF32 для Tensor Cores.")

    # Инициализация
    dataset = LocalImageDataset(r"..\data\images.parquet", CACHED_IMG_DIR, transform=train_transforms)

    # Оптимизированный DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Оставляем 4 для Windows
        pin_memory=True,  # Ускоряет перенос Тензоров CPU -> GPU
        persistent_workers=True  # Не пересоздает процессы каждую эпоху
    )

    # Модель и оптимизация
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    print("Начало быстрого обучения...")
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(dataloader, desc=f"Эпоха {epoch + 1}/{EPOCHS}")
        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # Освобождает память быстрее, чем zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

        epoch_loss = running_loss / len(dataset)
        print(f"Итог эпохи {epoch + 1}: Loss: {epoch_loss:.4f} | Точность: {100.0 * correct / total:.2f}%")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Модель сохранена в файл: {MODEL_SAVE_PATH}")

    print("Обучение завершено успешно!")


if __name__ == '__main__':
    main()
