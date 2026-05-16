"""Training loop. Designed for Colab: checkpoints to Drive every epoch so a
disconnect doesn't lose progress."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from . import config as C
from .model import parameter_groups, save_checkpoint


@dataclass
class History:
    train_loss: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    lrs: list = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = -1


def _epoch_pass(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    """One pass over `loader`. If `optimizer` is given, trains; else evaluates."""
    training = optimizer is not None
    model.train(training)

    total_loss, total_correct, total_n = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, leave=False, desc="train" if training else "val"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += x.size(0)

    return total_loss / total_n, total_correct / total_n


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    epochs: int = C.EPOCHS,
    lr_head: float = C.LR_HEAD,
    lr_backbone: float = C.LR_BACKBONE,
    weight_decay: float = C.WEIGHT_DECAY,
    label_smoothing: float = C.LABEL_SMOOTHING,
    models_dir: Optional[Path] = None,
    device: Optional[str] = None,
    tag: str = "mobilenetv2",
) -> History:
    """Train and return History. Saves:
      - {tag}_epoch{e}.pth  (per-epoch checkpoint, in case of disconnect)
      - {tag}_best.pth      (best val-accuracy weights)
      - {tag}_history.json  (loss/acc curves for the report)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(models_dir or C.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=label_smoothing
    )
    optimizer = torch.optim.AdamW(
        parameter_groups(model, lr_head, lr_backbone),
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    hist = History()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _epoch_pass(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = _epoch_pass(model, val_loader, criterion, device, None)
        scheduler.step()

        hist.train_loss.append(tr_loss); hist.train_acc.append(tr_acc)
        hist.val_loss.append(va_loss);   hist.val_acc.append(va_acc)
        hist.lrs.append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"| train loss {tr_loss:.4f} acc {tr_acc:.4f} "
            f"| val loss {va_loss:.4f} acc {va_acc:.4f} "
            f"| {time.time()-t0:.1f}s"
        )

        # Per-epoch checkpoint to Drive.
        save_checkpoint(
            model,
            models_dir / f"{tag}_epoch{epoch:02d}.pth",
            extra={"epoch": epoch, "val_acc": va_acc, "val_loss": va_loss},
        )

        if va_acc > hist.best_val_acc:
            hist.best_val_acc = va_acc
            hist.best_epoch = epoch
            save_checkpoint(
                model,
                models_dir / f"{tag}_best.pth",
                extra={"epoch": epoch, "val_acc": va_acc, "val_loss": va_loss},
            )

    # Persist history for plotting later / for the report.
    with open(models_dir / f"{tag}_history.json", "w") as f:
        json.dump(asdict(hist), f, indent=2)

    return hist
