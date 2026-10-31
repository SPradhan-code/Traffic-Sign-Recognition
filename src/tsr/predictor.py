"""Inference core shared by the API, web app and webcam app."""
import cv2
import numpy as np
from tensorflow import keras

from tsr.labels import CLASS_NAMES


class Predictor:
    def __init__(self, model_path: str, img_size: int = 32):
        self.model = keras.models.load_model(model_path, compile=False)
        self.img_size = img_size

    def _preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        img = cv2.resize(image_rgb, (self.img_size, self.img_size))
        return np.expand_dims(img.astype("float32"), axis=0)

    def predict(self, image_rgb: np.ndarray, top_k: int = 3) -> list:
        probs = self.model.predict(self._preprocess(image_rgb), verbose=0)[0]
        top = np.argsort(probs)[::-1][:top_k]
        return [
            {
                "class_id": int(i),
                "label": CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i),
                "confidence": float(probs[i]),
            }
            for i in top
        ]
