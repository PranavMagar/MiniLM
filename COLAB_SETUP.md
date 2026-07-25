# FinanceLM — Google Colab Training Guide

This guide walks you through running the FinanceLM training pipeline on
Google Colab using a GPU runtime.

---

## Prerequisites

- A Google account
- Google Colab (free tier works; Pro gives longer runtimes and better GPUs)
- The FinanceLM project files (see Step 1)

---

## Step 1 — Upload the Project to Colab

### Option A — Upload a ZIP

1. Zip the project on your local machine (exclude `__pycache__`, `.egg-info`):
   ```
   FinanceLM/
   ├── financelm/
   ├── scripts/
   ├── configs/
   ├── data/
   │   ├── minitext8/minitext8.txt
   │   ├── tokenizer/tokenizer.json
   │   └── processed/tokenized.pt
   ├── checkpoints/          (include if resuming)
   ├── requirements.txt
   └── pyproject.toml
   ```

2. In Colab, upload and extract:
   ```python
   from google.colab import files
   files.upload()           # select FinanceLM.zip
   
   import zipfile
   with zipfile.ZipFile("FinanceLM.zip", "r") as z:
       z.extractall("/content/FinanceLM")
   
   %cd /content/FinanceLM
   ```

### Option B — Clone from GitHub

```python
!git clone https://github.com/YOUR_USERNAME/FinanceLM.git /content/FinanceLM
%cd /content/FinanceLM
```

---

## Step 2 — Select a GPU Runtime

1. In Colab: **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **T4 GPU** (free) or **A100** (Pro)
3. Click **Save**

---

## Step 3 — Install Dependencies

```python
!pip install -r requirements.txt -q
!pip install -e . -q
```

---

## Step 4 — Verify the Environment

```python
!python scripts/colab_setup.py
```

Expected output:
```
  [OK]  Python version  (3.10+)
  [OK]  torch installed
  [OK]  GPU 0  (Tesla T4  15.8 GB VRAM)
  [OK]  AMP (mixed precision)  enabled for CUDA
  [OK]  MiniText8 corpus
  [OK]  Tokenizer
  [OK]  Processed dataset
  All checks passed ✓  —  ready to train!
```

---

## Step 5 — (Optional) Mount Google Drive

Mount Drive to persist checkpoints across sessions:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Create a checkpoint directory on Drive:

```python
import os
os.makedirs("/content/drive/MyDrive/FinanceLM/checkpoints", exist_ok=True)
```

Then edit `configs/training.yaml` to point to Drive:

```yaml
paths:
  checkpoint_dir: /content/drive/MyDrive/FinanceLM/checkpoints
```

Or set the environment variable before training:

```python
import os
os.environ["FINANCELM_CHECKPOINT_DIR"] = "/content/drive/MyDrive/FinanceLM/checkpoints"
```

---

## Step 6 — Start Training

```python
!python scripts/train.py
```

Or with custom settings:

```python
!python scripts/train.py --epochs 10 --batch-size 64
```

Recommended batch sizes by GPU:
| GPU | Batch size |
|-----|-----------|
| T4 (15 GB) | 32–64 |
| V100 (16 GB) | 64 |
| A100 (40 GB) | 128–256 |

---

## Step 7 — Resume Training

If your Colab session disconnects:

```python
!python scripts/train.py --resume checkpoints/latest.pt
```

With Drive checkpoints:

```python
!python scripts/train.py \
    --resume /content/drive/MyDrive/FinanceLM/checkpoints/latest.pt
```

---

## Step 8 — Generate Text

```python
!python scripts/generate.py --prompt "the history of" --max-tokens 200
```

Interactive mode (requires Colab input widget):

```python
!python scripts/generate.py
```

---

## Step 9 — Evaluate

```python
!python scripts/evaluate.py
```

With a specific checkpoint:

```python
!python scripts/evaluate.py --checkpoint checkpoints/best.pt --split val
```

---

## Step 10 — Save Checkpoints Back to Drive

After training:

```python
import shutil
shutil.copy(
    "/content/FinanceLM/checkpoints/best.pt",
    "/content/drive/MyDrive/FinanceLM/checkpoints/best.pt"
)
```

Or, if you configured `paths.checkpoint_dir` to point to Drive in Step 5,
checkpoints are saved there automatically.

---

## Configuration Reference

All training parameters are in `configs/training.yaml`.

Key values to tune for Colab:

| Parameter | Local default | Colab (T4) recommendation |
|---|---|---|
| `batch_size` | 8 | 32–64 |
| `num_workers` | 0 | 2 |
| `epochs` | 10 | 10–50 |
| `context_length` | 256 | 256–512 |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'financelm'`**
```python
!pip install -e . -q
```

**`CUDA out of memory`**
- Reduce `batch_size` in `configs/training.yaml` or pass `--batch-size 16`

**`Checkpoint not found`**
- Run `!python scripts/train.py` to create the first checkpoint

**`Processed dataset not found`**
- Either upload `data/processed/tokenized.pt`, or re-run:
  ```python
  !python scripts/prepare_dataset.py
  ```

**Session disconnected mid-training**
- Resume from last checkpoint:
  ```python
  !python scripts/train.py --resume checkpoints/latest.pt
  ```

**Drive not persisting between sessions**
- Re-mount Drive and set `paths.checkpoint_dir` in `training.yaml`

---

## File Transfer Checklist

Files you **must** upload to Colab:

```
✅ financelm/                    (entire package)
✅ scripts/                      (all scripts)
✅ configs/dataset.yaml
✅ configs/training.yaml
✅ requirements.txt
✅ pyproject.toml
✅ data/minitext8/minitext8.txt  (100 MB)
✅ data/tokenizer/tokenizer.json
✅ data/processed/tokenized.pt  (639 MB)
```

Files you **can optionally** include for resume:

```
⬜ checkpoints/latest.pt        (33 MB per checkpoint)
⬜ checkpoints/best.pt
```

Files to **exclude**:

```
❌ __pycache__/
❌ *.egg-info/
❌ .git/
❌ notebooks/
❌ tests/
❌ checkpoints/epoch_*.pt       (keep only latest + best for Colab)
```
