"""Inference wrapper. The UI imports this class and calls predict per
webcam frame. Handles both BGR arrays from OpenCV and RGB arrays from PIL
or MediaPipe via a constructor flag."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from . import config as C


ArrayLike = Union[np.ndarray, Image.Image]


class EmotionPredictor:
    """Stateful predictor. Build once, reuse for every frame."""

    def __init__(
        self,
        weights_path: Union[str, Path],
        input_is_bgr: bool = True,
        img_size: int = C.IMG_SIZE,
    ) -> None:
        self.model = tf.keras.models.load_model(str(weights_path))
        self.input_is_bgr = input_is_bgr
        self.img_size = img_size

    def _to_rgb_array(self, img: ArrayLike) -> np.ndarray:
        if isinstance(img, Image.Image):
            return np.asarray(img.convert("RGB"))
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Expected np.ndarray or PIL.Image, got {type(img)}")
        if img.ndim == 2:
            return np.stack([img] * 3, axis=-1)
        if img.ndim == 3 and img.shape[2] == 3:
            if self.input_is_bgr:
                img = img[:, :, ::-1]
            return np.ascontiguousarray(img)
        raise ValueError(f"Unsupported image shape: {img.shape}")

    def _prep(self, img: ArrayLike) -> np.ndarray:
        arr = self._to_rgb_array(img).astype(np.float32)
        resized = tf.image.resize(arr, (self.img_size, self.img_size)).numpy()
        return preprocess_input(resized)[None, ...]

    def predict(self, face_crop: ArrayLike) -> Dict:
        x = self._prep(face_crop)
        prob = self.model.predict(x, verbose=0)[0]
        idx = int(prob.argmax())
        return {
            "label": C.CLASSES[idx],
            "confidence": float(prob[idx]),
            "all_scores": {cls: float(p) for cls, p in zip(C.CLASSES, prob)},
        }

    def predict_batch(self, face_crops: list[ArrayLike]) -> list[Dict]:
        """Batched inference for callers that buffer frames."""
        if not face_crops:
            return []
        batch = np.concatenate([self._prep(c) for c in face_crops], axis=0)
        probs = self.model.predict(batch, verbose=0)
        out = []
        for prob in probs:
            idx = int(prob.argmax())
            out.append({
                "label": C.CLASSES[idx],
                "confidence": float(prob[idx]),
                "all_scores": {cls: float(p) for cls, p in zip(C.CLASSES, prob)},
            })
        return out
