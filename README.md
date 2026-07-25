# FinanceLM

A decoder-only GPT-style language model built from scratch in PyTorch.
The current phase uses **MiniText8** (100 MB Wikipedia text) to validate
the full training pipeline end-to-end before scaling to financial corpora.

---

## Architecture

| Component | Implementation |
|---|---|
| Embedding | Token embedding with weight tying to LM head |
| Position | Rotary Positional Embedding (RoPE) |
| Attention | Multi-head self-attention with causal mask |
| FFN | SwiGLU (gate × up → down projection) |
| Norm | RMSNorm (pre-norm, before each sub-layer) |
| Blocks | 8 × TransformerBlock |
| Parameters | ~10M |

---

## Project Structure

```
FinanceLM/
├── financelm/
│   ├── model/            # Architecture (attention, FFN, RoPE, RMSNorm …)
│   ├── data/             # Loader, preprocessing, dataset classes
│   ├── training/         # Trainer (AMP, grad clip) + CheckpointManager
│   ├── inference/        # Sampler (greedy / top-k / top-p) + Generator
│   └── paths.py          # Canonical path constants
│
├── scripts/
│   ├── download_minitext8.py  # Download the corpus
│   ├── inspect_dataset.py     # Print corpus statistics
│   ├── train_tokenizer.py     # Train BPE tokenizer
│   ├── prepare_dataset.py     # Tokenize + build sliding-window tensors
│   ├── verify_dataset.py      # Validate all pipeline artefacts
│   ├── train.py               # Training loop (resume supported)
│   ├── generate.py            # Interactive / single-prompt generation
│   └── evaluate.py            # Loss, perplexity, sample generations
│
├── configs/
│   ├── dataset.yaml           # Tokenizer vocab size, context length, stride
│   └── training.yaml          # Epochs, LR, warmup, checkpoint interval …
│
├── data/
│   ├── minitext8/             # Raw corpus (minitext8.txt)
│   ├── tokenizer/             # tokenizer.json
│   └── processed/             # tokenized.pt (sliding-window tensors)
│
├── checkpoints/               # Saved model checkpoints
├── requirements.txt
└── pyproject.toml
```

---

## Installation

```bash
conda activate financelm          # or: pip install -r requirements.txt
pip install -e .
```

Required packages: `torch`, `tokenizers`, `tqdm`, `numpy`, `pyyaml`

---

## Pipeline — Quick Start

Run these commands in order. Each step is independent and idempotent.

```bash
# 1. Download MiniText8 (100 MB, one-time)
python scripts/download_minitext8.py

# 2. Inspect the corpus
python scripts/inspect_dataset.py

# 3. Train the BPE tokenizer  (vocab_size from configs/dataset.yaml)
python scripts/train_tokenizer.py

# 4. Build the tokenized dataset  (sliding windows → tokenized.pt)
python scripts/prepare_dataset.py

# 5. Verify all artefacts
python scripts/verify_dataset.py

# 6. Train the model
python scripts/train.py

# 7. Generate text (interactive)
python scripts/generate.py

# 8. Evaluate
python scripts/evaluate.py
```

---

## Configuration

### `configs/dataset.yaml`

| Key | Default | Description |
|---|---|---|
| `tokenizer.vocab_size` | 8192 | BPE vocabulary size |
| `training.context_length` | 256 | Token window length |
| `training.stride` | 128 | Step between windows |
| `training.batch_size` | 8 | Batch size |

### `configs/training.yaml`

| Key | Default | Description |
|---|---|---|
| `training.epochs` | 10 | Total epochs |
| `training.learning_rate` | 3e-4 | Peak LR |
| `training.warmup_steps` | 1000 | Linear warmup steps |
| `training.weight_decay` | 0.1 | AdamW weight decay |
| `training.gradient_clip` | 1.0 | Gradient clipping norm |
| `training.val_fraction` | 0.05 | Fraction held out for validation |
| `checkpoint.save_every_steps` | 500 | Step-level checkpoint interval |
| `generation.max_new_tokens` | 200 | Default generation length |

---

## Training Commands

```bash
# Standard training
python scripts/train.py

# Resume from checkpoint
python scripts/train.py --resume checkpoints/latest.pt

# Override config values via CLI
python scripts/train.py --epochs 20 --batch-size 16 --lr 1e-4
```

---

## Generation Commands

```bash
# Interactive chat
python scripts/generate.py

# Single prompt
python scripts/generate.py --prompt "the stock market" --max-tokens 200

# From a specific checkpoint
python scripts/generate.py --checkpoint checkpoints/best.pt

# Change sampling strategy
python scripts/generate.py --strategy greedy
python scripts/generate.py --strategy top_k --top-k 50
python scripts/generate.py --temperature 0.5 --top-p 0.95 --seed 42
```

---

## Evaluation

```bash
# Evaluate latest checkpoint on full dataset
python scripts/evaluate.py

# Evaluate best checkpoint on validation split only
python scripts/evaluate.py --checkpoint checkpoints/best.pt --split val
```

---

## Checkpoint Resume

Checkpoints store: model weights, optimizer state, scheduler state,
AMP scaler state, Python/NumPy/PyTorch RNG states, epoch, global step,
and loss.

Resume is seamless:

```bash
python scripts/train.py --resume checkpoints/latest.pt
```

---

## Expected Outputs (MiniText8)

After 10 epochs on a CPU (~10–30 min depending on hardware):

| Metric | Expected range |
|---|---|
| Training loss | 4.5 – 6.0 |
| Validation loss | 4.8 – 6.5 |
| Perplexity | 120 – 600 |

These numbers validate the pipeline; quality improves significantly
with more data, longer context, and more epochs.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'financelm'`**
→ Run `pip install -e .` from the project root.

**`Tokenizer not found`**
→ Run `python scripts/train_tokenizer.py`.

**`Processed dataset not found`**
→ Run `python scripts/prepare_dataset.py`.

**`Checkpoint not found`**
→ Run `python scripts/train.py` first.

**CUDA out of memory**
→ Reduce `batch_size` in `configs/training.yaml` or pass `--batch-size 4`.

---

## Next Steps (before scaling to financial data)

1. Replace MiniText8 with a financial corpus (SEC filings, earnings calls, news).
2. Increase `vocab_size` to 32k–50k for financial terminology coverage.
3. Increase `context_length` to 1024–2048 and `embedding_dim` to 512+.
4. Add gradient accumulation for effective larger batch sizes on limited VRAM.
5. Implement `torch.compile` for faster training.
6. Add a learning-rate finder or sweep.
7. Add validation perplexity early-stopping.
