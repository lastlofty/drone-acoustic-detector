"""Генератор синтетического датасета — чтобы проверить весь пайплайн
БЕЗ реальных записей. Создаёт два класса:

  data/drone/      — гармоничное "жужжание" моторов с модуляцией
  data/background/ — шум, ветер, случайные тона

Это НЕ настоящий детектор дронов: на синтетике сеть учится отличать
искусственный буз от шума. Для рабочей модели замени эти папки
реальными записями (см. README).

Запуск:  python make_synthetic_data.py
"""
import os

import numpy as np
import soundfile as sf

import config

N_PER_CLASS = 60          # сколько клипов на класс
CLIP_SEC = config.CLIP_DURATION
SR = config.SAMPLE_RATE
rng = np.random.default_rng(42)


def _t():
    return np.linspace(0, CLIP_SEC, int(SR * CLIP_SEC), endpoint=False)


def drone_clip() -> np.ndarray:
    """Имитация звука дрона: основная частота мотора + гармоники + AM."""
    t = _t()
    f0 = rng.uniform(90, 220)          # частота вращения винтов
    sig = np.zeros_like(t)
    for h in range(1, 7):              # гармоники
        amp = 1.0 / h
        sig += amp * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))
    # амплитудная модуляция (биения винтов)
    am = 1.0 + 0.4 * np.sin(2 * np.pi * rng.uniform(5, 30) * t)
    sig *= am
    sig += 0.15 * rng.standard_normal(len(t))   # немного шума
    return _normalize(sig)


def background_clip() -> np.ndarray:
    """Фон: розовый шум + редкие случайные тона."""
    t = _t()
    noise = rng.standard_normal(len(t))
    # грубое приближение розового шума через скользящее среднее
    k = rng.integers(5, 40)
    noise = np.convolve(noise, np.ones(k) / k, mode="same")
    sig = noise
    if rng.random() < 0.5:             # иногда подмешиваем тон (птица/сигнал)
        f = rng.uniform(300, 4000)
        sig += 0.3 * np.sin(2 * np.pi * f * t)
    return _normalize(sig)


def _normalize(sig: np.ndarray) -> np.ndarray:
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    return (0.9 * sig).astype(np.float32)


def main():
    for cls, fn in (("drone", drone_clip), ("background", background_clip)):
        out_dir = os.path.join(config.DATA_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(N_PER_CLASS):
            sf.write(os.path.join(out_dir, f"{cls}_{i:03d}.wav"), fn(), SR)
        print(f"[+] {N_PER_CLASS} клипов -> {out_dir}")
    print("Готово. Теперь: python train.py")


if __name__ == "__main__":
    main()
