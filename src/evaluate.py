"""Честная оценка модели на отложенном test-наборе.

Считает precision / recall / F1, матрицу ошибок и ROC-AUC (дрон vs фон),
сохраняет графики и metrics.json в папку reports/.

Запуск:  python evaluate.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")  # без GUI
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

import config
from dataset import AudioFolderDataset, discover_classes, split_dataset
from model import DroneCNN

REPORTS_DIR = os.path.join(config.BASE_DIR, "reports")


def is_drone(name: str) -> bool:
    return name.lower() not in config.NEGATIVE_CLASSES


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    ckpt = torch.load(config.MODEL_PATH, map_location="cpu")
    classes = ckpt["classes"]
    split = tuple(ckpt.get("split", (0.7, 0.15, 0.15)))
    seed = ckpt.get("split_seed", 0)

    model = DroneCNN(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # тот же сплит, что и при обучении -> берём отложенный test
    ds = AudioFolderDataset(config.DATA_DIR, classes)
    _, _, test_ds = split_dataset(ds, split, seed)
    if len(test_ds) == 0:
        raise SystemExit("Test-набор пуст — добавь больше данных.")
    test_dl = DataLoader(test_ds, batch_size=16)

    y_true, y_pred, drone_score = [], [], []
    drone_idx = [i for i, c in enumerate(classes) if is_drone(c)]

    with torch.no_grad():
        for x, y in test_dl:
            probs = torch.softmax(model(x), dim=1).numpy()
            preds = probs.argmax(1)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(preds.tolist())
            # бинарный скор "это дрон" = сумма вероятностей дрон-классов
            drone_score.extend(probs[:, drone_idx].sum(axis=1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    drone_score = np.array(drone_score)

    # --- текстовый отчёт ---
    report = classification_report(
        y_true, y_pred, target_names=classes, digits=3, zero_division=0
    )
    print("\n=== Classification report (test) ===")
    print(report)

    # --- матрица ошибок ---
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Confusion matrix (test)")
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=130)
    plt.close()

    # --- ROC (дрон vs фон) ---
    y_true_bin = np.array([1 if is_drone(classes[t]) else 0 for t in y_true])
    roc_auc = None
    if 0 < y_true_bin.sum() < len(y_true_bin):  # есть оба класса
        fpr, tpr, _ = roc_curve(y_true_bin, drone_score)
        roc_auc = float(roc_auc_score(y_true_bin, drone_score))
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC — дрон vs фон (test)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(REPORTS_DIR, "roc_curve.png")
        plt.savefig(roc_path, dpi=130)
        plt.close()
        print(f"ROC-AUC (дрон vs фон): {roc_auc:.3f}")

    # --- сводка в JSON ---
    acc = float((y_true == y_pred).mean())
    metrics = {
        "test_samples": int(len(y_true)),
        "classes": classes,
        "accuracy": round(acc, 4),
        "roc_auc_drone_vs_bg": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nAccuracy (test): {acc:.3f}")
    print(f"Графики и metrics.json сохранены в: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
