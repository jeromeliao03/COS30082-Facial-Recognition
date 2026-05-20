"""Two-phase transfer-learning loop.

Phase one freezes the backbone and trains only the new classification head.
Phase two unfreezes the top portion of the backbone and continues training
at a much smaller learning rate. Per-epoch checkpoints save to Drive so a
Colab disconnect does not lose progress.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from . import config as C
from .model import unfreeze_top_n, count_trainable


def _compile(model, lr: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=C.LABEL_SMOOTHING
        ),
        metrics=["accuracy"],
    )


def _callbacks(models_dir: Path, tag: str, phase: str,
               best_threshold: float | None = None):
    """Per-phase callback list. best_threshold prevents phase 2 from
    overwriting phase 1's best checkpoint with a temporarily worse model
    while newly-unfrozen layers settle."""
    best_kwargs = {}
    if best_threshold is not None:
        best_kwargs["initial_value_threshold"] = best_threshold
    return [
        ModelCheckpoint(
            filepath=str(models_dir / f"{tag}_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
            **best_kwargs,
        ),
        ModelCheckpoint(
            filepath=str(models_dir / f"{tag}_{phase}_epoch{{epoch:02d}}.keras"),
            save_best_only=False,
            verbose=0,
        ),
        CSVLogger(str(models_dir / f"{tag}_{phase}_log.csv"), append=False),
    ]


def _merge_histories(h1, h2) -> dict:
    return {k: list(h1.history[k]) + list(h2.history[k]) for k in h1.history}


def train(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_data,
    val_data,
    models_dir: Path | None = None,
    tag: str = "mobilenetv2",
):
    """Run both phases. Returns a merged history dict."""
    models_dir = Path(models_dir or C.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: head only.
    print(f"\nPhase 1: feature extraction (head only, lr={C.LR_PHASE1})")
    base_model.trainable = False
    _compile(model, lr=C.LR_PHASE1)
    trainable, total = count_trainable(model)
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")

    t0 = time.time()
    h1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=C.EPOCHS_PHASE1,
        callbacks=_callbacks(models_dir, tag, phase="p1"),
        verbose=1,
    )
    p1_time = time.time() - t0
    print(f"Phase 1 finished in {p1_time:.1f}s")
    phase1_best_val_acc = max(h1.history["val_accuracy"])

    # Phase 2: fine-tune backbone tail.
    print(f"\nPhase 2: fine-tuning (unfreeze last {C.UNFREEZE_LAST_N_LAYERS} "
          f"layers, lr={C.LR_PHASE2})")
    fine_tune_at = unfreeze_top_n(base_model, C.UNFREEZE_LAST_N_LAYERS)
    print(f"Freezing layers 0..{fine_tune_at}, "
          f"training layers {fine_tune_at}..{len(base_model.layers)}")
    _compile(model, lr=C.LR_PHASE2)
    trainable, total = count_trainable(model)
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")

    t0 = time.time()
    h2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=C.EPOCHS_PHASE1 + C.EPOCHS_PHASE2,
        initial_epoch=C.EPOCHS_PHASE1,
        callbacks=_callbacks(models_dir, tag, phase="p2",
                             best_threshold=phase1_best_val_acc),
        verbose=1,
    )
    p2_time = time.time() - t0
    print(f"Phase 2 finished in {p2_time:.1f}s")

    merged = _merge_histories(h1, h2)
    history_path = models_dir / f"{tag}_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "history": merged,
            "phase1_epochs": C.EPOCHS_PHASE1,
            "phase2_epochs": C.EPOCHS_PHASE2,
            "phase1_time_s": p1_time,
            "phase2_time_s": p2_time,
        }, f, indent=2)
    print(f"\nHistory saved to {history_path}")

    return merged
