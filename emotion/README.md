# Emotion Detection Module — MobileNetV2 on FER-2013

**Author:** Hasindu 

## What this module does

Takes a cropped face image and returns one of 7 emotions (`angry, disgust, fear, happy, neutral, sad, surprise`) with a confidence score. Designed to be called per-frame by the attendance system's UI.

```python
from emotion.src.predict import EmotionPredictor
predictor = EmotionPredictor("models/mobilenetv2_best.pth")
predictor.predict(face_crop_bgr)
# -> {"label": "happy", "confidence": 0.87, "all_scores": {...}}
```

## Folder layout

```
emotion/
├── README.md                        # this file
├── requirements.txt
├── .gitignore                       # excludes data/, *.pth
├── notebooks/
│   └── emotion_mobilenetv2.ipynb    # Colab driver — run this
├── src/
│   ├── config.py                    # paths, hyperparameters
│   ├── data.py                      # FER-2013 loaders + augmentations
│   ├── model.py                     # MobileNetV2 builder (transfer learning)
│   ├── train.py                     # training loop, per-epoch Drive checkpoints
│   ├── evaluate.py                  # accuracy, F1, confusion matrix, plots
│   └── predict.py                   # EmotionPredictor (UI integration point)
├── models/                          # gitignored; populated by training
└── reports/                         # gitignored; populated by evaluation (plots, JSON)
```

## Why this layout (modules + one notebook, not one mega-notebook)

- **UI teammate imports** `EmotionPredictor` directly — no copy-pasting cells.
- **Sanija mirrors** this structure for ResNet18: swap `model.py`, keep everything else. → fair comparison.
- **Git diffs** stay readable (logic is in `.py`, not buried in notebook JSON).
- The **notebook** is still the runtime in Colab and produces the plots the group report needs.

## How to run (Colab)

1. **One-time setup**
   - Create folder `MyDrive/MLGroup/emotion/` in Google Drive.
   - Upload your `kaggle.json` (Kaggle → Account → API → Create New Token) to `MyDrive/MLGroup/kaggle.json`.
2. **Open the notebook** `emotion/notebooks/emotion_mobilenetv2.ipynb` in Colab. Runtime → Change runtime type → **GPU (T4)**.
3. **Run cells top to bottom.** First run downloads FER-2013 from Kaggle to Drive (~60 MB, ~1 min). Subsequent runs skip the download.
4. **Training** (~30–60 min on T4 for 25 epochs). Per-epoch checkpoints saved to Drive as `mobilenetv2_epoch{NN}.pth`. The best one (by val accuracy) is duplicated as `mobilenetv2_best.pth`.
5. **Evaluation cells** produce `confusion_matrix.png`, `training_curves.png`, and `test_metrics.json` in `MyDrive/MLGroup/emotion/reports/`.

If Colab disconnects mid-training, just rerun — the cell `train(...)` starts fresh, but per-epoch checkpoints from the previous run are still on Drive if you need them.

## Design decisions (interview talking points)

| Choice | Why |
|---|---|
| **MobileNetV2** over ResNet18 | ~3.4M params vs 11.7M, ~6× fewer FLOPs. Live demo runs face-recognition + anti-spoof + emotion every frame — speed matters. ImageNet top-1 is also slightly better. |
| **Transfer learning, freeze early blocks** | Project rules forbid fully pre-trained models without further training, but allow partial fine-tuning. Last 4 inverted-residual blocks + new head are trainable; everything else is frozen. |
| **224×224 RGB input** | What ImageNet-pretrained MobileNetV2 expects. Grayscale 48×48 FER images are upsampled and channel-triplicated. |
| **ImageNet normalisation stats** | Pretrained weights expect `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`. Using `[0.5]*3` would silently degrade accuracy. |
| **Class-weighted cross-entropy** | FER-2013's `disgust` class is ~13× rarer than `happy`. Without weighting, the model learns to ignore it. |
| **Label smoothing 0.05** | FER-2013 labels are noisy (well documented); slight smoothing prevents overconfidence. |
| **Stratified train/val split** | Random split could underrepresent `disgust` in val. We split per-class. |
| **Two-LR optimiser** | New head gets `1e-3`, unfrozen backbone gets `1e-4`. Standard transfer-learning practice. |
| **AdamW + cosine LR** | Reliable defaults; no babysitting needed for a one-shot Colab run. |
| **Save every epoch** | Colab disconnects after ~12h. Drive persistence is the only way to avoid losing a long run. |

## Expected performance

FER-2013 SOTA is ~75% test accuracy. This recipe typically lands at **65–70% test accuracy / ~0.60 macro-F1**. The two famous confusion pairs to discuss in the report:

- `fear ↔ surprise` (eyes wide, mouth open in both)
- `sad ↔ neutral` (subtle expressions, often ambiguous)

## Comparison with ResNet18 (Sanija's model)

Once both models are trained, compare on the **same test set with the same metrics** (accuracy, macro-F1, per-class F1). Pick the winner for the live demo. If MobileNetV2 is within 1–2 points of ResNet18, prefer MobileNetV2 for inference latency.

## Future extensions (already factored in)

- `EmotionPredictor.predict_batch` handles batched inference if the UI buffers frames.
- The `EmotionPredictor.input_is_bgr` flag handles OpenCV (BGR) vs MediaPipe/PIL (RGB) — avoids the classic colour-space-swap bug.
- Adding a new dataset (RAF-DB, AffectNet) means changing only `config.CLASSES` and the data root.
