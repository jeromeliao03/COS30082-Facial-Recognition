"""Grad-CAM for the MobileNetV2 emotion classifier.

Visualises where on the face the model's evidence for its predicted class
came from. Implemented from scratch with tf.GradientTape rather than a
third-party library so the mechanism is auditable end-to-end.

Algorithm: take the gradient of the predicted class's score with respect
to the activations of the last convolutional feature map; weight each
channel of those activations by the spatial mean of its gradient; sum the
weighted activations across channels; ReLU the result; normalise to
[0, 1]; upsample to face-crop size; render as a JET colourmap and blend
with the original crop.

Usage:
    gradcam = GradCAM(predictor.model)
    heatmap, pred_class, prob = gradcam.compute(face_bgr)
    overlaid = gradcam.overlay(face_bgr, heatmap)
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from . import config as C


class GradCAM:
    """Stateful Grad-CAM. Build once, call .compute() per face."""

    def __init__(
        self,
        model: tf.keras.Model,
        img_size: int = C.IMG_SIZE,
        input_is_bgr: bool = True,
    ) -> None:
        self.model = model
        self.img_size = img_size
        self.input_is_bgr = input_is_bgr

        # The full model is: Input -> base_model -> GAP -> Dropout -> Dense.
        # Locate the nested base_model and the head layers we need.
        self._base = self._find_base_model(model)
        self._gap = next(
            l for l in model.layers
            if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)
        )
        self._dense = next(
            l for l in model.layers
            if isinstance(l, tf.keras.layers.Dense)
        )

    @staticmethod
    def _find_base_model(model: tf.keras.Model) -> tf.keras.Model:
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                return layer
        raise ValueError("No nested base model found inside the loaded model.")

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        face = face_bgr[:, :, ::-1] if self.input_is_bgr else face_bgr
        face = np.ascontiguousarray(face).astype(np.float32)
        resized = tf.image.resize(face, (self.img_size, self.img_size)).numpy()
        return preprocess_input(resized)[None, ...]

    def compute(
        self,
        face_bgr: np.ndarray,
        pred_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Return (heatmap, pred_class, confidence).

        heatmap is a 2D float array in [0, 1] at the base model's output
        resolution (typically 4x4 for 128 input). overlay() upsamples it
        back to the face crop size.

        If pred_class is None, uses the model's own argmax as the target.
        """
        x = tf.convert_to_tensor(self._preprocess(face_bgr))

        # include_top=False, so base(x) returns the final conv feature map
        # (out_relu activations). We watch it explicitly so the tape can
        # differentiate the class score with respect to it.
        with tf.GradientTape() as tape:
            conv_out = self._base(x, training=False)
            tape.watch(conv_out)
            pooled = self._gap(conv_out)
            preds = self._dense(pooled)
            if pred_class is None:
                pred_class = int(tf.argmax(preds[0]).numpy())
            class_score = preds[:, pred_class]

        grads = tape.gradient(class_score, conv_out)              # (1, h, w, c)
        channel_weights = tf.reduce_mean(grads, axis=(0, 1, 2))   # (c,)

        conv_np = conv_out[0].numpy()                             # (h, w, c)
        weights_np = channel_weights.numpy()                      # (c,)

        heatmap = np.tensordot(conv_np, weights_np, axes=[[2], [0]])  # (h, w)
        heatmap = np.maximum(heatmap, 0)
        max_val = float(heatmap.max())
        if max_val > 0:
            heatmap = heatmap / max_val

        prob = float(preds[0, pred_class].numpy())
        return heatmap.astype(np.float32), pred_class, prob

    def overlay(
        self,
        face_bgr: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Render heatmap as a JET colourmap and blend with the original crop."""
        import cv2
        h, w = face_bgr.shape[:2]
        hm = cv2.resize(heatmap, (w, h))
        hm_u8 = np.uint8(255 * np.clip(hm, 0.0, 1.0))
        coloured = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        return cv2.addWeighted(coloured, alpha, face_bgr, 1.0 - alpha, 0)
