"""Глобальная конфигурация проекта.

Пути считаются относительно корня проекта, поэтому скрипты можно
запускать из папки src/ (python train.py и т.д.).
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Аудио ---
SAMPLE_RATE = 22050            # частота дискретизации, Гц
CLIP_DURATION = 1.0            # длина одного анализируемого окна, сек
SAMPLES_PER_CLIP = int(SAMPLE_RATE * CLIP_DURATION)

# --- Признаки (лог-мел-спектрограмма) ---
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

# --- Пути ---
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "drone_cnn.pt")

# --- Детекция ---
DETECT_THRESHOLD = 0.6         # порог вероятности для тревоги
SMOOTH_WINDOW = 5              # окно сглаживания вероятностей (кол-во кадров)
# имена классов, которые считаются "не дрон" (фон/тишина)
NEGATIVE_CLASSES = {"background", "noise", "silence", "other"}
