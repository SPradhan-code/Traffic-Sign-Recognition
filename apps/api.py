"""REST API:  uvicorn apps.api:app --reload   (docs at /docs)"""
import os
import sys
from functools import lru_cache
from pathlib import Path

# Ensure project root and src/ are in sys.path
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402

from tsr.config import load_config  # noqa: E402
from tsr.predictor import Predictor  # noqa: E402

app = FastAPI(title="Traffic Sign Recognition API", version="1.0.0")
MAX_BYTES = 5 * 1024 * 1024


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Model is loaded once, on first use (override path with TSR_MODEL_PATH)."""
    cfg = load_config()
    return Predictor(os.environ.get("TSR_MODEL_PATH", cfg["model_path"]), cfg["img_size"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "Only JPEG/PNG images are supported")
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 5 MB)")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return {"predictions": get_predictor().predict(img, top_k=max(1, min(top_k, 10)))}
