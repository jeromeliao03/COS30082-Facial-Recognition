import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


class SpoofExplainability:
    """
    Explainable Anti-Spoofing using Grad-CAM.

    This module uses the existing spoof_model.keras and generates
    a heatmap showing which face regions influenced the spoof decision.
    """

    def __init__(self, model_path="models/spoof_model.keras"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path, compile=False)

        self.input_size = self._get_input_size()
        self.base_model = self._find_nested_cnn_model()
        self.last_conv_layer_name = self._find_last_conv_layer_name(self.base_model)

        print("Spoof explainability model loaded.")
        print("Model input size:", self.input_size)
        print("Nested CNN model:", self.base_model.name)
        print("Last conv layer:", self.last_conv_layer_name)

        self.grad_model = self._build_grad_model()

    def _get_input_size(self):
        input_shape = self.model.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        height = input_shape[1]
        width = input_shape[2]

        return (width, height)

    def _find_nested_cnn_model(self):
        for layer in self.model.layers:
            if isinstance(layer, keras.Model):
                return layer

        raise ValueError("No nested CNN model found inside spoof model.")

    def _find_last_conv_layer_name(self, model):
        for layer in reversed(model.layers):
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
                return layer.name

        raise ValueError("No Conv2D or DepthwiseConv2D layer found in nested CNN model.")

    def _build_grad_model(self):
        model_input = self.model.input

        last_conv_layer = self.base_model.get_layer(self.last_conv_layer_name)

        base_grad_model = keras.models.Model(
            inputs=self.base_model.input,
            outputs=[
                last_conv_layer.output,
                self.base_model.output
            ]
        )

        conv_output, x = base_grad_model(model_input)

        start_passing = False

        for layer in self.model.layers:
            if layer == self.base_model:
                start_passing = True
                continue

            if start_passing:
                x = layer(x)

        return keras.models.Model(
            inputs=model_input,
            outputs=[conv_output, x]
        )

    def preprocess_face(self, face_bgr):
        face_resized = cv2.resize(face_bgr, self.input_size)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        face_array = face_rgb.astype("float32") / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        return face_array

    def predict_spoof_score(self, face_bgr):
        face_array = self.preprocess_face(face_bgr)
        prediction = self.model.predict(face_array, verbose=0)

        return float(np.ravel(prediction)[0])

    def generate_gradcam(self, face_bgr):
        face_array = self.preprocess_face(face_bgr)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(face_array)

            spoof_score = predictions[:, 0]
            loss = spoof_score

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            raise ValueError("Gradients could not be computed for spoof model.")

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)

        max_value = tf.reduce_max(heatmap)

        if max_value != 0:
            heatmap = heatmap / max_value

        heatmap = heatmap.numpy()

        overlay = self.create_overlay(face_bgr, heatmap)

        score = float(np.ravel(predictions.numpy())[0])

        return score, heatmap, overlay

    def create_overlay(self, face_bgr, heatmap, alpha=0.45):
        """
        Creates coloured heatmap overlay on original face crop.
        Handles dtype safely for OpenCV.
        """

        face = face_bgr.copy()

        if face.dtype != np.uint8:
            if face.max() <= 1.0:
                face = face * 255.0

            face = np.clip(face, 0, 255).astype(np.uint8)

        heatmap = np.nan_to_num(heatmap)
        heatmap = np.clip(heatmap, 0, 1)

        heatmap_resized = cv2.resize(
            heatmap,
            (face.shape[1], face.shape[0])
        )

        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        heatmap_color = heatmap_color.astype(np.uint8)
        face = face.astype(np.uint8)

        overlay = cv2.addWeighted(
            face,
            1 - alpha,
            heatmap_color,
            alpha,
            0
        )

        return overlay

    def explain_face(self, face_bgr):
        score, heatmap, overlay = self.generate_gradcam(face_bgr)

        if score >= 0.5:
            label = "SPOOF"
            confidence = score
        else:
            label = "REAL"
            confidence = 1 - score

        return {
            "score": score,
            "label": label,
            "confidence": confidence,
            "heatmap": heatmap,
            "overlay": overlay
        }