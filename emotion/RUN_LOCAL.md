# Run locally

Trains the emotion model on own machine instead of Colab/Kaggle.

## Requirements

- A Kaggle API token (`kaggle.json`)

## Setup

```bash
git clone -b emotion-branch https://github.com/jeromeliao03/COS30082-Facial-Recognition.git
cd COS30082-Facial-Recognition

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r emotion/requirements.txt
```

Check the GPU is visible:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If it prints `[]`, fix that before training. Otherwise training will run on CPU and take hours.

## Get the dataset

Put the Kaggle token at `~/.kaggle/kaggle.json`, then:

```bash
mkdir -p ~/fer2013
kaggle datasets download -d msambare/fer2013 -p ~/fer2013 --unzip
```

Should see `train/` and `test/` folders under `~/fer2013`, each containing 7 emotion subfolders.

## Run training

```bash
python emotion/run_local.py
```

Two-phase training, 25 epochs total. Takes 15–60 min depending on the GPU. Per-epoch checkpoints save to `~/emotion_outputs/models/`.

To put the dataset or outputs somewhere else:

```bash
EMOTION_DATA_DIR=/path/to/fer2013 EMOTION_OUTPUT_DIR=/path/to/outputs python emotion/run_local.py
```

## What you get

In `~/emotion_outputs/`:

- `models/mobilenetv2_best.keras` — trained model
- `models/mobilenetv2_history.json` — training curves data
- `models/mobilenetv2_p1_log.csv`, `mobilenetv2_p2_log.csv` — per-phase logs
- `models/mobilenetv2_p1_epoch*.keras`, `mobilenetv2_p2_epoch*.keras` — per-epoch backups
- `reports/training_curves.png`
- `reports/confusion_matrix.png`
- `reports/test_metrics.json`

## If something breaks

- `ModuleNotFoundError: tensorflow` — virtualenv not activated. Run `source .venv/bin/activate`.
- `FER-2013 not found at ...` — dataset path wrong. Check `~/fer2013/train/happy/` exists.
- Step time > 2 s in training — running on CPU. Kill it, fix GPU visibility.
- WSL `nvidia-smi: command not found` — Windows driver too old, update to 525+.
