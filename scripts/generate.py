"""
scripts/generate.py
====================
Interactive text generation with a trained FinanceLM checkpoint.

Usage
-----
    python scripts/generate.py
    python scripts/generate.py --checkpoint checkpoints/best.pt
    python scripts/generate.py --prompt "the stock market" --max-tokens 200
    python scripts/generate.py --strategy greedy
    python scripts/generate.py --temperature 0.7 --top-p 0.95 --seed 42

Arguments
---------
    --checkpoint PATH     Checkpoint to load (default: checkpoints/latest.pt).
    --prompt TEXT         Single prompt (non-interactive).
    --max-tokens N        Maximum new tokens to generate (default: 200).
    --strategy STRATEGY   Sampling strategy: greedy | top_k | top_p (default: top_p).
    --temperature FLOAT   Sampling temperature (default: 0.8).
    --top-k N             Top-K value (default: 40).
    --top-p FLOAT         Nucleus-p value (default: 0.9).
    --seed N              Random seed for reproducibility (optional).
    --log-level LEVEL     Logging verbosity (default: WARNING).
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from financelm.inference.generator import Generator
from financelm.inference.sampling import SamplingStrategy
from financelm.model.config import ModelConfig
from financelm.model.model import FinanceLM
from financelm.paths import CHECKPOINT_DIR, TOKENIZER_FILE
from financelm.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate")

_STRATEGY_MAP: dict[str, SamplingStrategy] = {
    "greedy": SamplingStrategy.GREEDY,
    "top_k":  SamplingStrategy.TOP_K,
    "top_p":  SamplingStrategy.TOP_P,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate text with a trained FinanceLM checkpoint."
    )
    p.add_argument("--checkpoint",   type=str,   default=None,
                   help="Checkpoint path (default: checkpoints/latest.pt).")
    p.add_argument("--prompt",       type=str,   default=None,
                   help="Single prompt for non-interactive mode.")
    p.add_argument("--max-tokens",   type=int,   default=200)
    p.add_argument("--strategy",     type=str,   default="top_p",
                   choices=list(_STRATEGY_MAP))
    p.add_argument("--temperature",  type=float, default=0.8)
    p.add_argument("--top-k",        type=int,   default=40)
    p.add_argument("--top-p",        type=float, default=0.9)
    p.add_argument("--seed",         type=int,   default=None,
                   help="Set random seed for reproducible generation.")
    p.add_argument("--log-level",    default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # ── locate checkpoint ─────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (
        CHECKPOINT_DIR / "latest.pt"
    )

    manager = CheckpointManager()
    if not ckpt_path.exists():
        print(f"\nCheckpoint not found: {ckpt_path}")
        print("Run:  python scripts/train.py")
        sys.exit(1)

    if not TOKENIZER_FILE.exists():
        print(f"\nTokenizer not found: {TOKENIZER_FILE}")
        print("Run:  python scripts/train_tokenizer.py")
        sys.exit(1)

    # ── model ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig()
    model  = FinanceLM(config).to(device)

    ckpt = manager.load(
        checkpoint_path=ckpt_path,
        model=model,
        device=device,
        restore_rng=args.seed is None,
    )
    print(
        f"\nLoaded  {ckpt_path.name}"
        f"  (epoch={ckpt.get('epoch', '?')}"
        f"  loss={ckpt.get('loss', float('nan')):.4f})"
    )

    strategy = _STRATEGY_MAP[args.strategy]
    generator = Generator(
        model=model,
        tokenizer_path=TOKENIZER_FILE,
        device=device,
        max_seq_length=config.max_seq_length,
    )

    # ── single-prompt mode ────────────────────────────────────────────
    if args.prompt:
        result = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            strategy=strategy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(f"\n{result}\n")
        return

    # ── interactive mode ──────────────────────────────────────────────
    print("\n==============================")
    print(" FinanceLM  —  Text Generation")
    print(f" strategy={args.strategy}  T={args.temperature}"
          f"  top_k={args.top_k}  top_p={args.top_p}")
    print(" Type 'exit' to quit")
    print("==============================\n")

    while True:
        try:
            prompt = input("You : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if prompt.lower() in ("exit", "quit"):
            break
        if not prompt:
            continue

        result = generator.generate(
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            strategy=strategy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(f"AI  : {result}\n")


if __name__ == "__main__":
    main()
