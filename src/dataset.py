"""Датасет: папки внутри data/ = классы. Длинные файлы режутся на окна."""
import os
import glob

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

import config
from features import wav_to_logmel

AUDIO_EXTS = ("*.wav", "*.flac", "*.ogg", "*.mp3")


def discover_classes(data_dir: str = config.DATA_DIR):
    """Список классов = имена подпапок в data/ (отсортированы)."""
    if not os.path.isdir(data_dir):
        return []
    return sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )


class AudioFolderDataset(Dataset):
    """Каждый сэмпл — окно CLIP_DURATION секунд из аудиофайла."""

    def __init__(self, data_dir: str, classes):
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.index = []  # список (path, start_sample, label)

        for c in classes:
            label = self.class_to_idx[c]
            files = []
            for ext in AUDIO_EXTS:
                files.extend(glob.glob(os.path.join(data_dir, c, ext)))
            for f in files:
                try:
                    dur = librosa.get_duration(path=f)
                except Exception:
                    continue
                n_windows = max(1, int(dur // config.CLIP_DURATION))
                for k in range(n_windows):
                    self.index.append((f, k * config.SAMPLES_PER_CLIP, label))

        # детерминированный порядок -> воспроизводимые train/val/test сплиты
        self.index.sort()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, start, label = self.index[idx]
        offset = start / config.SAMPLE_RATE
        y, _ = librosa.load(
            path, sr=config.SAMPLE_RATE, mono=True,
            offset=offset, duration=config.CLIP_DURATION,
        )
        feat = wav_to_logmel(y)
        x = torch.from_numpy(feat).unsqueeze(0)  # (1, N_MELS, frames)
        return x, label


def split_dataset(ds, fractions=(0.7, 0.15, 0.15), seed: int = 0):
    """Детерминированный train/val/test сплит (один и тот же в train.py и
    evaluate.py при одинаковом seed), чтобы метрики считались на честном
    отложенном test-наборе."""
    n = len(ds)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    n_test = n - n_train - n_val
    return random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
