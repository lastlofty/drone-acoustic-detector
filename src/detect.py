"""Детекция в реальном времени с микрофона.

Скользящее окно 1 сек -> спектрограмма -> CNN -> вероятности классов.
Если вероятность "дрон"-класса превышает порог — тревога.

Запуск:  python detect.py
Выход:   Ctrl+C
"""
import queue
import sys

import numpy as np
import sounddevice as sd
import torch

import config
from features import wav_to_logmel
from model import DroneCNN
from smoothing import ProbSmoother

BLOCK_SEC = 0.25  # как часто обновляем оценку


def load_model():
    ckpt = torch.load(config.MODEL_PATH, map_location="cpu")
    classes = ckpt["classes"]
    model = DroneCNN(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes


def is_drone_class(name: str) -> bool:
    return name.lower() not in config.NEGATIVE_CLASSES


def render(probs, classes):
    """Однострочный 'дашборд' в терминале."""
    parts = []
    drone_p = 0.0
    for name, p in zip(classes, probs):
        parts.append(f"{name}:{p:4.0%}")
        if is_drone_class(name):
            drone_p = max(drone_p, p)
    bar_len = 20
    filled = int(drone_p * bar_len)
    bar = "#" * filled + "-" * (bar_len - filled)
    alert = "  <<< ДРОН!" if drone_p >= config.DETECT_THRESHOLD else ""
    line = f"\r[{bar}] " + "  ".join(parts) + alert + " " * 6
    sys.stdout.write(line)
    sys.stdout.flush()


def main():
    try:
        model, classes = load_model()
    except FileNotFoundError:
        raise SystemExit("Модель не найдена. Сначала: python train.py")
    print(f"Классы: {classes}. Слушаю микрофон (Ctrl+C — выход)\n")

    q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata[:, 0].copy())

    buf = np.zeros(config.SAMPLES_PER_CLIP, dtype=np.float32)
    blocksize = int(config.SAMPLE_RATE * BLOCK_SEC)
    smoother = ProbSmoother(config.SMOOTH_WINDOW)

    with sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=1,
        blocksize=blocksize,
        callback=callback,
    ):
        try:
            while True:
                block = q.get()
                buf = np.concatenate([buf, block])[-config.SAMPLES_PER_CLIP:]
                feat = wav_to_logmel(buf)
                x = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1)[0].numpy()
                probs = smoother.update(probs)   # сглаживание во времени
                render(probs, classes)
        except KeyboardInterrupt:
            print("\nОстановлено.")


if __name__ == "__main__":
    main()
