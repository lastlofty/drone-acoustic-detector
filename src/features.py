"""Извлечение признаков: аудио -> нормализованная лог-мел-спектрограмма."""
import numpy as np
import librosa

import config


def fix_length(y: np.ndarray) -> np.ndarray:
    """Приводит сигнал к фиксированной длине SAMPLES_PER_CLIP."""
    n = config.SAMPLES_PER_CLIP
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    else:
        y = y[:n]
    return y


def wav_to_logmel(y: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Сигнал -> лог-мел-спектрограмма, нормированная в [0, 1].

    Возвращает массив формы (N_MELS, frames), dtype float32.
    """
    y = fix_length(y)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-9)
    return logmel.astype(np.float32)


def load_audio(path: str, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Загружает аудиофайл как моно-сигнал нужной частоты."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y
