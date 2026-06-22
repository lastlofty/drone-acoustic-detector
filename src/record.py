"""Помощник для сбора РЕАЛЬНОГО датасета с микрофона.

Запуск:
    python record.py <класс> <секунды> [количество_файлов]

Примеры:
    python record.py drone 5 10        # 10 файлов по 5 сек в data/drone/
    python record.py background 5 10    # фон

Записывай дрон с разных расстояний/ракурсов, фон — в разной обстановке.
"""
import os
import sys
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

import config


def record_one(seconds: float) -> np.ndarray:
    audio = sd.rec(
        int(seconds * config.SAMPLE_RATE),
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio[:, 0]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    cls = sys.argv[1]
    seconds = float(sys.argv[2])
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    out_dir = os.path.join(config.DATA_DIR, cls)
    os.makedirs(out_dir, exist_ok=True)
    existing = len(os.listdir(out_dir))

    print(f"Класс '{cls}': запишу {count} файлов по {seconds} сек.")
    for i in range(count):
        input(f"  [{i + 1}/{count}] Enter — старт записи...")
        print("  ● запись...")
        audio = record_one(seconds)
        path = os.path.join(out_dir, f"{cls}_{existing + i:03d}.wav")
        sf.write(path, audio, config.SAMPLE_RATE)
        print(f"  -> {path}")
        time.sleep(0.2)

    print("Готово.")


if __name__ == "__main__":
    main()
