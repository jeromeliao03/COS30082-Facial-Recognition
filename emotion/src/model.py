"""MobileNetV2 with a seven-class classification head.

Built with the Keras Functional API. The base model is called with
training=False so batch-normalisation layers stay in inference mode and
their pretrained running statistics are preserved during fine-tuning.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from . import config as C


def build_mobilenetv2(
    img_size: int = C.IMG_SIZE,
    num_classes: int = C.NUM_CLASSES,
    dropout: float = C.DROPOUT,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Return (full_model, base_model). base_model is exposed so the
    training code can freeze and unfreeze its layers between phases."""
    img_shape = (img_size, img_size, 3)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_emotion")
    return model, base_model


def unfreeze_top_n(base_model: tf.keras.Model, n_layers: int) -> int:
    """Unfreeze the last n layers of the base model in place. Returns the
    index at which fine-tuning starts."""
    base_model.trainable = True
    total = len(base_model.layers)
    fine_tune_at = max(0, total - n_layers)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    return fine_tune_at


def count_trainable(model: tf.keras.Model) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    total = sum(tf.size(w).numpy() for w in model.weights)
    return int(trainable), int(total)
