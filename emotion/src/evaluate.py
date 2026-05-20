"""Evaluation metrics and plotting helpers. All metric maths uses sklearn
so the numbers are framework agnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

from . import config as C


def collect_predictions(model: tf.keras.Model, data_iterator):
    """Run the model over an iterator and return (y_true, y_pred, y_prob).
    The iterator must have shuffle=False so labels and predictions stay
    aligned."""
    y_prob = model.predict(data_iterator, verbose=1)
    y_pred = y_prob.argmax(axis=1)
    y_true = data_iterator.classes
    return y_true, y_pred, y_prob


def metrics_report(y_true, y_pred) -> dict:
    """JSON-serialisable metrics dict plus a printable text report."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "per_class_f1": f1_score(y_true, y_pred, average=None).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=C.CLASSES, digits=4
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(C.NUM_CLASSES))
        ).tolist(),
    }


def plot_confusion_matrix(cm, save_path: Optional[Path] = None,
                          normalize: bool = True):
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f" if normalize else ".0f",
        cmap="Blues", xticklabels=C.CLASSES, yticklabels=C.CLASSES, ax=ax,
        cbar_kws={"label": "fraction" if normalize else "count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        "FER-2013 confusion matrix"
        + (" (row-normalised)" if normalize else "")
    )
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_training_curves(history: dict, phase1_epochs: int = 0,
                         save_path: Optional[Path] = None):
    """Two-panel loss and accuracy plot. If phase1_epochs is given, a
    vertical line marks the boundary between phases."""
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(epochs, history["accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()

    if phase1_epochs > 0:
        for ax in axes:
            ax.axvline(phase1_epochs + 0.5, linestyle="--", color="grey",
                       linewidth=1, label="phase 1 to phase 2")
        axes[0].legend(); axes[1].legend()

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig
