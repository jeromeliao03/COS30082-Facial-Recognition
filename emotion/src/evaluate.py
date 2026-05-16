"""Evaluation: accuracy, per-class precision/recall/F1, and confusion matrix.

These outputs feed directly into the group report's Results & Discussion."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)

from . import config as C


@torch.no_grad()
def collect_predictions(model, loader, device: Optional[str] = None):
    """Run the model over `loader`, return (y_true, y_pred, y_prob)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    all_true, all_pred, all_prob = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        pred = prob.argmax(axis=1)
        all_true.append(y.numpy())
        all_pred.append(pred)
        all_prob.append(prob)

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


def metrics_report(y_true, y_pred) -> dict:
    """Numbers + a printable text report."""
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


def plot_confusion_matrix(cm, save_path: Optional[Path] = None, normalize: bool = True):
    """Plot a confusion matrix. Returns the matplotlib Figure."""
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
    ax.set_title("FER-2013 confusion matrix" + (" (row-normalised)" if normalize else ""))
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_training_curves(history: dict, save_path: Optional[Path] = None):
    """Two-panel: loss and accuracy. Reads the dict produced by train.History."""
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig
