"""Видеодетектор дронов на YOLO — визуальный близнец акустического.

Детекция в реальном времени с веб-камеры или из видеофайла: рамки вокруг
дронов + уверенность + баннер тревоги. Вместе с акустикой это пара
"звук + зрение" — мини-система sensor fusion.

Запуск:
    python detect_video.py                      # веб-камера (источник 0)
    python detect_video.py --source clip.mp4    # из файла
    python detect_video.py --weights ../models/yolo_drone.pt

ВАЖНО про веса: для реальной детекции нужны веса, обученные на дронах
(см. train_video.py + датасет, напр. с Roboflow). Если своих весов нет,
берётся предобученная YOLOv8n (COCO) — она НЕ знает класс 'drone' и нужна
лишь чтобы убедиться, что пайплайн запускается.
"""
import argparse
import os

import cv2
from ultralytics import YOLO

# имена классов, которые считаем дроном (в твоих обученных весах)
DRONE_NAMES = {"drone", "uav", "quadcopter", "drones"}
DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "yolo_drone.pt",
)


def parse_args():
    p = argparse.ArgumentParser(description="YOLO drone video detector")
    p.add_argument("--source", default="0",
                   help="индекс веб-камеры (0) или путь к видео")
    p.add_argument("--weights", default=None,
                   help="путь к .pt весам (по умолч. models/yolo_drone.pt)")
    p.add_argument("--conf", type=float, default=0.35,
                   help="порог уверенности")
    return p.parse_args()


def resolve_weights(weights) -> str:
    if weights:
        return weights
    if os.path.exists(DEFAULT_WEIGHTS):
        return DEFAULT_WEIGHTS
    print("[!] Свои веса не найдены — беру предобученную YOLOv8n (COCO).")
    print("[!] Она не знает класс 'drone'. Обучи свои: см. train_video.py")
    return "yolov8n.pt"


def main():
    args = parse_args()
    model = YOLO(resolve_weights(args.weights))
    names = model.names  # id -> имя класса

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Не открыть источник: {args.source}")

    print("Видеодетектор запущен. 'q' — выход.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model(frame, conf=args.conf, verbose=False)[0]

        drone_hit = False
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = names.get(cls_id, str(cls_id))
            is_drone = name.lower() in DRONE_NAMES
            if is_drone:
                drone_hit = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 60, 255) if is_drone else (120, 120, 120)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        banner = "DRONE DETECTED" if drone_hit else "CLEAR"
        bcolor = (0, 60, 255) if drone_hit else (60, 200, 120)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (15, 20, 30), -1)
        cv2.putText(frame, banner, (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, bcolor, 2)

        cv2.imshow("Acoustic Drone Detector — video twin", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
