"""
classification_model.py  (TensorFlow / Keras baseline — Hasi branch)
--------------------------------------------------------------------
Mirrors the PyTorch teammate's src/classification_model.py.

Architecture (Section 2.1 — supervised / classification-based approach):
    Input  → augmentation (train-only)
           → ResNet50 backbone (frozen except last N layers) with global avg pool
           → Dense(EMBEDDING_DIM) + BatchNorm + ReLU      ← embedding layer
           → Dense(num_people)                             ← softmax classifier

For verification (Section 2.2/2.3) the softmax head is ignored — embeddings
are taken from the layer BEFORE the classifier and compared with cosine /
Euclidean similarity.

build_classification_model() returns two Keras Models that share weights:
    classifier_model : used for training (loss = sparse categorical CE)
    embedding_model  : used at eval time to extract face embeddings
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from config import (
    IMAGE_SIZE, EMBEDDING_DIM, CLF_UNFREEZE_LAYERS, BACKBONE_NAME,
)


# ---------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------
def _build_backbone():
    """Return a Keras backbone that maps (224, 224, 3) → (feature_dim,)."""
    if BACKBONE_NAME == "ResNet50":
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(*IMAGE_SIZE, 3),
            pooling="avg",   # global average pool → 2048-D vector
        )
        return base

    if BACKBONE_NAME == "ResNet18":
        # Requires:  pip install image-classifiers
        try:
            from classification_models.tfkeras import Classifiers
        except ImportError as e:
            raise ImportError(
                "BACKBONE_NAME='ResNet18' needs the `image-classifiers` package.\n"
                "Install it in Colab with:  !pip install image-classifiers"
            ) from e
        ResNet18, _preprocess = Classifiers.get("resnet18")
        base_no_pool = ResNet18(
            include_top=False,
            weights="imagenet",
            input_shape=(*IMAGE_SIZE, 3),
        )
        # Wrap with global avg pool to mimic the include_top=False + pooling='avg' API
        inp = layers.Input(shape=(*IMAGE_SIZE, 3))
        x = base_no_pool(inp)
        x = layers.GlobalAveragePooling2D()(x)
        return Model(inputs=inp, outputs=x, name="resnet18_backbone")

    raise ValueError(f"Unknown BACKBONE_NAME: {BACKBONE_NAME}")


# ---------------------------------------------------------------
# Augmentation block
# ---------------------------------------------------------------
def _build_augmentation():
    """
    Mirrors the PyTorch train-time pipeline:
        RandomHorizontalFlip(0.5)
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        RandomRotation(±15°)

    Saturation jitter is approximated via a Lambda since Keras has no
    built-in RandomSaturation layer.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(15.0 / 360.0, fill_mode="reflect"),
            layers.RandomBrightness(factor=0.2, value_range=(-1.0, 1.0)),
            layers.RandomContrast(factor=0.2),
            layers.Lambda(_random_saturation, name="random_saturation"),
        ],
        name="augmentation",
    )


def _random_saturation(x):
    """Random saturation applied per-image. Active during training only."""
    # Inputs are in [-1, 1]; tf.image.random_saturation expects [0, 1].
    x01 = (x + 1.0) / 2.0
    x01 = tf.image.random_saturation(x01, lower=0.8, upper=1.2)
    return tf.clip_by_value(x01 * 2.0 - 1.0, -1.0, 1.0)


# ---------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------
def build_classification_model(num_people):
    """
    Build the classification-baseline model.

    Args:
        num_people : int  number of identity classes in the training set

    Returns:
        classifier_model : tf.keras.Model  (image) → logits
        embedding_model  : tf.keras.Model  (image) → 512-D embedding
                           Shares weights with classifier_model — train one,
                           use either to predict.
    """
    backbone = _build_backbone()

    # Freeze everything, then unfreeze the last N layers of the backbone.
    backbone.trainable = True
    for layer in backbone.layers[:-CLF_UNFREEZE_LAYERS]:
        layer.trainable = False
    for layer in backbone.layers[-CLF_UNFREEZE_LAYERS:]:
        layer.trainable = True

    augmentation = _build_augmentation()

    # Functional API so we can expose the embedding as a separate Model
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = augmentation(inputs)                        # train-only ops
    features = backbone(x)                           # (batch, feature_dim)

    embedding = layers.Dense(EMBEDDING_DIM, name="embedding_dense")(features)
    embedding = layers.BatchNormalization(name="embedding_bn")(embedding)
    embedding = layers.ReLU(name="embedding_relu")(embedding)

    logits = layers.Dense(num_people, name="classifier")(embedding)

    classifier_model = Model(inputs=inputs, outputs=logits,
                             name="classification_model")
    embedding_model  = Model(inputs=inputs, outputs=embedding,
                             name="embedding_extractor")

    trainable = sum(int(tf.size(v)) for v in classifier_model.trainable_variables)
    total     = classifier_model.count_params()

    print(f"\nClassification Model ready (TensorFlow / Keras)")
    print(f"  Backbone        : {BACKBONE_NAME}")
    print(f"  People          : {num_people}")
    print(f"  Embedding dim   : {EMBEDDING_DIM}")
    print(f"  Layers unfrozen : last {CLF_UNFREEZE_LAYERS} backbone layers "
          f"(of {len(backbone.layers)})")
    print(f"  Trainable       : {trainable:,} / {total:,} params "
          f"({100 * trainable / total:.1f}% unfrozen)")

    return classifier_model, embedding_model
