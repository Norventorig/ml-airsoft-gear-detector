# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms, models
#
#
# def main():
#     # 1. Настройка GPU (RTX 3060)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Используемое устройство: {device}")
#     if device.type == 'cuda':
#         print(f"Видеокарта: {torch.cuda.get_device_name(0)}")
#
#     # 2. Пути к данным и параметры
#     data_dir =r'..\data\dataset'
#     num_classes = 11
#     batch_size = 32
#     num_epochs = 10
#     learning_rate = 0.001
#
#     # 3. Трансформация изображений (изменение размера, нормализация)
#     data_transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#
#     # 4. Загрузка датасета из папок
#     dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
#
#     # Проверка количества классов
#     actual_classes = len(dataset.classes)
#     print(f"Обнаружено классов в папке: {actual_classes}")
#     if actual_classes != num_classes:
#         print(f"Внимание: Найдено {actual_classes} папок вместо {num_classes}!")
#
#     # Разделение на обучение (80%) и валидацию (20%)
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
#
#     # Создание загрузчиков данных
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
#
#     # 5. Инициализация предобученной модели (ResNet18)
#     model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#
#     # Изменяем последний слой под 11 классов
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, num_classes)
#
#     # Перенос модели на RTX 3060
#     model = model.to(device)
#
#     # 6. Функция потерь и оптимизатор
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#     # 7. Цикл обучения
#     print("Начало обучения...")
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#
#         for images, labels in train_loader:
#             # Перенос данных на GPU
#             images, labels = images.to(device), labels.to(device)
#
#             # Прямой шаг
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#
#             # Обратный шаг
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item() * images.size(0)
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
#
#         epoch_loss = running_loss / len(train_dataset)
#         epoch_acc = (correct / total) * 100
#         print(f"Эпоха [{epoch + 1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Точность (Train): {epoch_acc:.2f}%")
#
#     print("Обучение завершено!")
#
#     # 8. Сохранение весов модели
#     save_path = "best_model_11_classes.pth"
#     torch.save(model.state_dict(), save_path)
#     print(f"Модель успешно сохранена в файл: {save_path}")
#
#     # Сохранение соответствия классов (какая папка какому индексу равна)
#     print("Соответствие классов и индексов:")
#     for class_name, class_idx in dataset.class_to_idx.items():
#         print(f"Класс {class_idx}: {class_name}")
#
#
# if __name__ == '__main__':
#     main()
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def main():
    # 1. Настройка GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    data_dir = r'..\data\dataset'
    num_classes = 11
    batch_size = 32
    num_epochs = 15  # Немного увеличили, так как с аугментацией модель учится дольше, но качественнее
    learning_rate = 0.0005  # Снизили шаг для более стабильного дообучения

    # 2. Аугментация для тренировки + базовая обработка для валидации
    train_transforms = transforms.Compose([
        transforms.Resize((240, 240)),  # Чуть больше размер перед случайной обрезкой
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # Отражение по горизонтали
        transforms.RandomRotation(degrees=15),  # Случайный поворот на +/- 15 градусов
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # Изменение яркости и контраста
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Загрузка общего датасета (сначала без трансформаций, чтобы правильно разделить)
    full_dataset = datasets.ImageFolder(root=data_dir)

    # Индексы для разделения
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_idx, val_idx = torch.utils.data.random_split(range(len(full_dataset)), [train_size, val_size])

    # Применяем разные трансформации к обучающей и валидационной выборкам
    train_dataset = torch.utils.data.Subset(datasets.ImageFolder(root=data_dir, transform=train_transforms), train_idx)
    val_dataset = torch.utils.data.Subset(datasets.ImageFolder(root=data_dir, transform=val_transforms), val_idx)

    # Создание загрузчиков
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 4. Инициализация модели ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Начало обучения и валидации...")
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # --- ФАЗА ОБУЧЕНИЯ ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        epoch_train_loss = train_loss / len(train_idx)
        epoch_train_acc = (train_correct / train_total) * 100

        # --- ФАЗА ВАЛИДАЦИИ ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():  # Отключаем расчет градиентов для скорости и экономии памяти
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        epoch_val_loss = val_loss / len(val_idx)
        epoch_val_acc = (val_correct / val_total) * 100

        print(f"Эпоха [{epoch + 1}/{num_epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%")

        # Сохраняем модель, только если она показала лучшую точность на ВАЛИДАЦИИ
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), r"models\subcategory_image\best_model_11_classes.pth")
            print("--> Сохранена новая лучшая модель по метрике Val Accuracy!")

    print(f"\nОбучение успешно завершено! Лучшая точность на валидации: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
