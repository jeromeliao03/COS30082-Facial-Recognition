"""FER-2013 data generators using Keras ImageDataGenerator.

preprocess_input rescales pixels to [-1, 1], which is the range MobileNetV2
was pretrained on. color_mode='rgb' triplicates FER's grayscale channel to
match the pretrained backbone's three-channel input.
"""

from __future__ import annotations

from pathlib import Path

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from . import config as C


def _train_aug_generator() -> ImageDataGenerator:
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=C.VAL_SPLIT,
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        brightness_range=(0.85, 1.15),
    )


def _eval_generator() -> ImageDataGenerator:
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=C.VAL_SPLIT,
    )


def build_dataloaders(
    train_dir: Path | None = None,
    test_dir: Path | None = None,
    img_size: int = C.IMG_SIZE,
    batch_size: int = C.BATCH_SIZE,
):
    """Return train, validation, and test iterators.

    Validation uses no augmentation. Test uses the held-out test folder
    with neither augmentation nor any train/val split.
    """
    train_dir = str(train_dir or C.TRAIN_DIR)
    test_dir = str(test_dir or C.TEST_DIR)

    train_data = _train_aug_generator().flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=C.SEED,
    )

    val_data = _eval_generator().flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=C.SEED,
    )

    test_data = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    ).flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    # Class order on disk must match config.CLASSES.
    assert list(train_data.class_indices.keys()) == C.CLASSES, (
        f"Class order mismatch.\n  on disk: {list(train_data.class_indices.keys())}\n"
        f"  config : {C.CLASSES}"
    )

    return train_data, val_data, test_data
