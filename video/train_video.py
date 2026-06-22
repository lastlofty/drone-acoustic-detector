"""Дообучение YOLO на датасете дронов (формат Ultralytics/YOLOv8).

Как получить датасет:
    1. Открой Roboflow Universe (universe.roboflow.com) и найди датасет
       по запросу "drone detection" (есть несколько публичных).
    2. Export -> формат "YOLOv8" -> скачаешь папку с data.yaml,
       images/ и labels/.

Обучение:
    python train_video.py --data path/to/data.yaml --epochs 50

Результат: runs/detect/train/weights/best.pt
Скопируй его в ../models/yolo_drone.pt — и detect_video.py подхватит.
"""
import argparse
import shutil

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLO on drones")
    p.add_argument("--data", required=True, help="путь к data.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--base", default="yolov8n.pt",
                   help="базовые веса (n/s/m — компромисс скорость/точность)")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.base)
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)

    best = results.save_dir / "weights" / "best.pt"
    print(f"\nЛучшие веса: {best}")
    if best.exists():
        dst = "../models/yolo_drone.pt"
        shutil.copy(best, dst)
        print(f"Скопировано в {dst} — теперь: python detect_video.py")


if __name__ == "__main__":
    main()
