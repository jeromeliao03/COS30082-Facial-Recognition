"""Inference wrapper. This is the ONLY public surface the UI teammate needs.

Usage from the UI code:
    from emotion.src.predict import EmotionPredictor
    predictor = EmotionPredictor("path/to/mobilenetv2_best.pth")
    result = predictor.predict(face_bgr)   # numpy HxWx3 BGR (OpenCV) or RGB PIL
    # -> {"label": "happy", "confidence": 0.87,
    #     "all_scores": {"angry": 0.01, ..., "surprise": 0.02}}
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from PIL import Image

from . import config as C
from .data import inference_tf
from .model import load_checkpoint


ArrayLike = Union[np.ndarray, Image.Image]


class EmotionPredictor:
    """Stateful predictor. Build once, reuse for every webcam frame."""

    def __init__(
        self,
        weights_path: Union[str, Path],
        device: str | None = None,
        input_is_bgr: bool = True,
    ) -> None:
        """
        Args:
            weights_path: path to a checkpoint saved by train.py.
            device: "cuda", "cpu" or None (auto).
            input_is_bgr: True if face crops come from OpenCV (default), False
                if they come from PIL / Mediapipe (RGB).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_checkpoint(weights_path, map_location=self.device)
        self.model.to(self.device).eval()
        self.transform = inference_tf()
        self.input_is_bgr = input_is_bgr

    # ------------------------------------------------------------------ utils
    def _to_pil(self, img: ArrayLike) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Expected np.ndarray or PIL.Image, got {type(img)}")
        if img.ndim == 2:
            return Image.fromarray(img).convert("RGB")
        if img.ndim == 3 and img.shape[2] == 3:
            if self.input_is_bgr:
                img = img[:, :, ::-1]              # BGR -> RGB
            return Image.fromarray(np.ascontiguousarray(img)).convert("RGB")
        raise ValueError(f"Unsupported image shape: {img.shape}")

    # ----------------------------------------------------------------- public
    @torch.no_grad()
    def predict(self, face_crop: ArrayLike) -> Dict:
        """Predict emotion for a single cropped face."""
        pil = self._to_pil(face_crop)
        x = self.transform(pil).unsqueeze(0).to(self.device)
        prob = torch.softmax(self.model(x), dim=1).squeeze(0).cpu().numpy()
        idx = int(prob.argmax())
        return {
            "label": C.CLASSES[idx],
            "confidence": float(prob[idx]),
            "all_scores": {cls: float(p) for cls, p in zip(C.CLASSES, prob)},
        }

    @torch.no_grad()
    def predict_batch(self, face_crops: list[ArrayLike]) -> list[Dict]:
        """Predict for a batch - useful if the UI buffers frames."""
        if not face_crops:
            return []
        tensors = torch.stack([self.transform(self._to_pil(c)) for c in face_crops])
        tensors = tensors.to(self.device)
        probs = torch.softmax(self.model(tensors), dim=1).cpu().numpy()
        out = []
        for prob in probs:
            idx = int(prob.argmax())
            out.append({
                "label": C.CLASSES[idx],
                "confidence": float(prob[idx]),
                "all_scores": {cls: float(p) for cls, p in zip(C.CLASSES, prob)},
            })
        return out
