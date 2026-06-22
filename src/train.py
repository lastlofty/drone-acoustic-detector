"""Обучение CNN на спектрограммах из data/. Сохраняет модель + список классов.

Запуск:  python train.py
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import config
from dataset import AudioFolderDataset, discover_classes
from model import DroneCNN

EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-3
VAL_FRACTION = 0.2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = discover_classes(config.DATA_DIR)
    if len(classes) < 2:
        raise SystemExit(
            "Нужно минимум 2 класса (подпапки в data/). "
            "Запусти 'python make_synthetic_data.py' или добавь записи."
        )
    print(f"Классы: {classes}  |  устройство: {device}")

    ds = AudioFolderDataset(config.DATA_DIR, classes)
    if len(ds) < 4:
        raise SystemExit("Слишком мало данных. Добавь аудио в data/.")

    n_val = max(1, int(len(ds) * VAL_FRACTION))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = DroneCNN(len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / n_train

        # валидация
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
        acc = correct / n_val
        print(f"epoch {epoch:02d}  loss={train_loss:.3f}  val_acc={acc:.3f}")

        if acc >= best_acc:
            best_acc = acc
            torch.save(
                {"state_dict": model.state_dict(), "classes": classes},
                config.MODEL_PATH,
            )

    print(f"\nЛучшая val_acc={best_acc:.3f}. Модель: {config.MODEL_PATH}")
    print("Теперь запусти детектор:  python detect.py")


if __name__ == "__main__":
    main()
