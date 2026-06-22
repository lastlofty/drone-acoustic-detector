"""Веб-дашборд детектора в реальном времени (FastAPI + WebSocket).

Микрофон слушается в фоновом потоке, оценки и спектрограмма пушатся в
браузер по вебсокету. Открой http://127.0.0.1:8000

Запуск:
    python -m uvicorn webapp:app --reload
    (или просто: python webapp.py)
"""
import asyncio
import json
import os
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

import config
from features import wav_to_logmel
from model import DroneCNN
from smoothing import ProbSmoother

BLOCK_SEC = 0.25
STATIC_DIR = os.path.join(config.BASE_DIR, "src", "static")


def load_model():
    ckpt = torch.load(config.MODEL_PATH, map_location="cpu")
    classes = ckpt["classes"]
    model = DroneCNN(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes


model, CLASSES = load_model()

# Последнее состояние, разделяемое между аудио-потоком и вебсокетами.
_state = {
    "probs": [0.0] * len(CLASSES),
    "drone_p": 0.0,
    "mel": [],
    "ts": 0.0,
}
_lock = threading.Lock()


def _is_drone(name: str) -> bool:
    return name.lower() not in config.NEGATIVE_CLASSES


def audio_worker():
    """Фоновый поток: читает микрофон, считает оценки, обновляет _state."""
    q: queue.Queue = queue.Queue()

    def cb(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    buf = np.zeros(config.SAMPLES_PER_CLIP, dtype=np.float32)
    blocksize = int(config.SAMPLE_RATE * BLOCK_SEC)
    smoother = ProbSmoother(config.SMOOTH_WINDOW)

    with sd.InputStream(
        samplerate=config.SAMPLE_RATE, channels=1,
        blocksize=blocksize, callback=cb,
    ):
        while True:
            block = q.get()
            buf = np.concatenate([buf, block])[-config.SAMPLES_PER_CLIP:]
            feat = wav_to_logmel(buf)
            x = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].numpy()
            probs = smoother.update(probs)
            drone_p = max(
                (p for n, p in zip(CLASSES, probs) if _is_drone(n)),
                default=0.0,
            )
            with _lock:
                _state["probs"] = [float(p) for p in probs]
                _state["drone_p"] = float(drone_p)
                _state["mel"] = np.round(feat, 3).tolist()
                _state["ts"] = time.time()


app = FastAPI(title="Acoustic Drone Detector")


@app.on_event("startup")
def _startup():
    threading.Thread(target=audio_worker, daemon=True).start()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with _lock:
                payload = {
                    "classes": CLASSES,
                    "probs": _state["probs"],
                    "drone_p": _state["drone_p"],
                    "mel": _state["mel"],
                    "threshold": config.DETECT_THRESHOLD,
                }
            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass


# Статика монтируется последней, чтобы не перехватывать /ws.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webapp:app", host="127.0.0.1", port=8000, reload=False)
