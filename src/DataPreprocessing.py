"""
DataPreprocessing.py  (TensorFlow / Keras baseline — Hasi branch)
-----------------------------------------------------------------
tf.data pipelines that mirror the PyTorch teammate's src/DataPreprocessing.py:

  - build_classification_datasets()  → train + val tf.data.Dataset of (image, label)
  - build_verification_dataset()     → tf.data.Dataset of (image1, image2, label)
  - build_triplet_dataset()          → tf.data.Dataset of (anchor, positive, negative)

Image normalisation matches the PyTorch baseline exactly: pixels mapped
to [-1, 1]. Augmentation is intentionally *not* applied here — it lives
inside the model as Keras Random* layers (see classification_model.py).
That's the modern Keras pattern and runs the augmentation on the GPU.
"""

import os
import random

import numpy as np
import tensorflow as tf

from config import (
    IMAGE_SIZE,
    TRAIN_DIR, VAL_DIR, VERIFICATION_DIR, VERIFICATION_PAIRS,
    NORMALISE_MEAN, NORMALISE_STD,
    CLF_BATCH_SIZE, TRIPLET_BATCH_SIZE, TRIPLET_PER_EPOCH,
    AUTOTUNE, SEED,
)


# ---------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------
def _decode_and_resize(path):
    """Read an image file, decode to RGB float32, resize to IMAGE_SIZE."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    return tf.cast(img, tf.float32)


def _normalise(img):
    """Map [0, 255] → [-1, 1] (matches PyTorch baseline's mean=0.5/std=0.5)."""
    return (img - NORMALISE_MEAN) / NORMALISE_STD


def _load_image(path, label):
    img = _normalise(_decode_and_resize(path))
    return img, label


# ---------------------------------------------------------------
# Folder scanning (analogous to PyTorch TrainLoader)
# ---------------------------------------------------------------
def _scan_folder(folder, person_to_id=None):
    """
    Walk a classification folder of the form:
        folder/<person_id>/<photo>.jpg

    Returns:
        paths        : list[str]   absolute image paths
        labels       : list[int]   integer class id per image
        person_to_id : dict[str,int]  reusable mapping (used by val to align with train)
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}\n"
            f"Set DATA_ROOT in config.py to the location of your dataset."
        )

    people = sorted(
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d))
    )
    if person_to_id is None:
        person_to_id = {p: i for i, p in enumerate(people)}

    paths, labels = [], []
    for person in people:
        if person not in person_to_id:
            # Person seen at val time but not at train time — skip (closed-set classifier).
            continue
        pid = person_to_id[person]
        pdir = os.path.join(folder, person)
        for fname in os.listdir(pdir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(pdir, fname))
                labels.append(pid)
    return paths, labels, person_to_id


# ---------------------------------------------------------------
# Public dataset builders
# ---------------------------------------------------------------
def build_classification_datasets():
    """
    Returns:
        train_ds   : tf.data.Dataset  yielding (image, label) batches
        val_ds     : tf.data.Dataset  yielding (image, label) batches
        num_people : int              number of identity classes
    """
    train_paths, train_labels, person_to_id = _scan_folder(TRAIN_DIR)
    val_paths, val_labels, _ = _scan_folder(VAL_DIR, person_to_id=person_to_id)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .shuffle(buffer_size=len(train_paths), seed=SEED, reshuffle_each_iteration=True)
        .map(_load_image, num_parallel_calls=AUTOTUNE)
        .batch(CLF_BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        .map(_load_image, num_parallel_calls=AUTOTUNE)
        .batch(CLF_BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    print(f"\nClassification datasets ready")
    print(f"  Training people : {len(person_to_id)}")
    print(f"  Training photos : {len(train_paths)}")
    print(f"  Val photos      : {len(val_paths)}")

    return train_ds, val_ds, len(person_to_id)


def build_verification_dataset(batch_size=32):
    """
    Reads verification_pairs_val.txt and returns a tf.data.Dataset of
    (image1, image2, label) — same protocol the PyTorch baseline uses
    so AUC is computed over the identical pair set.
    """
    if not os.path.exists(VERIFICATION_PAIRS):
        raise FileNotFoundError(
            f"Verification pairs file not found: {VERIFICATION_PAIRS}"
        )

    pairs = []
    with open(VERIFICATION_PAIRS, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                p1, p2, label = parts
                pairs.append((_resolve_pair_path(p1),
                              _resolve_pair_path(p2),
                              int(label)))

    paths1 = [p[0] for p in pairs]
    paths2 = [p[1] for p in pairs]
    labels = [p[2] for p in pairs]

    same = sum(1 for l in labels if l == 1)
    diff = sum(1 for l in labels if l == 0)
    print(f"Verification pairs: {len(pairs)} loaded "
          f"({same} same person, {diff} different people)")

    def _load_pair(p1, p2, lbl):
        img1 = _normalise(_decode_and_resize(p1))
        img2 = _normalise(_decode_and_resize(p2))
        return img1, img2, lbl

    ds = (
        tf.data.Dataset.from_tensor_slices((paths1, paths2, labels))
        .map(_load_pair, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return ds


def _resolve_pair_path(name):
    """Accept either '0001.jpg' or 'verification_data/0001.jpg' style entries."""
    n = name.replace("\\", "/")
    if n.startswith("verification_data/"):
        n = n.split("/", 1)[1]
    return os.path.join(VERIFICATION_DIR, n)


# ---------------------------------------------------------------
# Triplet sampler (kept here for parity; only needed for metric learning)
# ---------------------------------------------------------------
def build_triplet_dataset(num_triplets=TRIPLET_PER_EPOCH):
    """
    Returns a tf.data.Dataset that yields (anchor, positive, negative) image
    triplets — anchor and positive are different photos of the same person,
    negative is a photo of a different person. Mirrors the PyTorch
    TripletLoader random sampler.
    """
    paths, labels, _ = _scan_folder(TRAIN_DIR)

    by_person = {}
    for path, lbl in zip(paths, labels):
        by_person.setdefault(lbl, []).append(path)

    usable = [pid for pid, ps in by_person.items() if len(ps) >= 2]
    skipped = len(by_person) - len(usable)
    if len(usable) < 2:
        raise ValueError(
            "TripletLoader requires at least 2 people with >=2 photos each."
        )
    print(f"TripletLoader: {len(usable)} people usable "
          f"({skipped} skipped — only 1 photo), "
          f"{num_triplets} triplets per epoch")

    rng = random.Random(SEED)

    def _generator():
        for _ in range(num_triplets):
            anchor_pid = rng.choice(usable)
            neg_pid = rng.choice([p for p in usable if p != anchor_pid])
            anchor_path, positive_path = rng.sample(by_person[anchor_pid], 2)
            negative_path = rng.choice(by_person[neg_pid])
            yield anchor_path, positive_path, negative_path

    def _load_triplet(a, p, n):
        return (
            _normalise(_decode_and_resize(a)),
            _normalise(_decode_and_resize(p)),
            _normalise(_decode_and_resize(n)),
        )

    ds = (
        tf.data.Dataset.from_generator(
            _generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.string),
            ),
        )
        .map(_load_triplet, num_parallel_calls=AUTOTUNE)
        .batch(TRIPLET_BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    return ds
