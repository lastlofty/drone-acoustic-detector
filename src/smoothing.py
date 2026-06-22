"""Сглаживание вероятностей во времени — чтобы тревога не 'дрожала'
от кадра к кадру. Скользящее среднее по последним N оценкам.
"""
from collections import deque

import numpy as np


class ProbSmoother:
    def __init__(self, window: int = 5):
        self.buf = deque(maxlen=max(1, window))

    def update(self, probs) -> np.ndarray:
        """Добавляет вектор вероятностей и возвращает усреднённый."""
        self.buf.append(np.asarray(probs, dtype=np.float64))
        return np.mean(self.buf, axis=0)

    def reset(self):
        self.buf.clear()
