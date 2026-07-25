"""
financelm.data.tokenizer_dataset
=================================
Two Dataset implementations for FinanceLM training:

TokenizerDataset
    Tokenizes the corpus on first use, builds sliding windows in memory.
    Good for quick iteration; requires the raw corpus and tokenizer.

ProcessedDataset
    Loads the pre-built ``tokenized.pt`` produced by ``prepare_dataset.py``.
    Avoids re-tokenizing on every run — preferred for repeated training runs.

Both implement the standard PyTorch Dataset interface and produce
(input_ids, target_ids) pairs for next-token-prediction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tokenizers import Tokenizer

from financelm.data.loader import load_minitext8
from financelm.data.preprocessing import preprocess

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TokenizerDataset — tokenizes corpus on-the-fly
# ---------------------------------------------------------------------------

class TokenizerDataset(Dataset):
    """
    Sliding-window next-token-prediction dataset.

    Tokenizes the corpus once at construction time, then provides
    (input_ids, target_ids) windows on demand.

    Parameters
    ----------
    corpus_path:
        Path to the plain-text corpus file.
    tokenizer_path:
        Path to the saved HuggingFace ``tokenizers`` JSON file.
    context_length:
        Number of tokens per input window.
    stride:
        Step size between consecutive windows.
    """

    def __init__(
        self,
        corpus_path: Path,
        tokenizer_path: Path,
        context_length: int,
        stride: int,
    ) -> None:
        self.context_length = context_length
        self.stride = stride

        raw  = load_minitext8(Path(corpus_path))
        text = preprocess(raw)

        tokenizer_path = Path(tokenizer_path).resolve()
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {tokenizer_path}\n"
                "Run:  python scripts/train_tokenizer.py"
            )
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokens: list[int] = tokenizer.encode(text).ids

        if len(self.tokens) < context_length + 1:
            raise ValueError(
                f"Corpus too small after tokenization.\n"
                f"  Tokens         : {len(self.tokens)}\n"
                f"  context_length : {context_length}\n"
                "Reduce context_length or use a larger corpus."
            )

        max_start = len(self.tokens) - context_length - 1
        self._starts: list[int] = list(range(0, max_start + 1, stride))

        logger.info(
            "TokenizerDataset: tokens=%d  windows=%d  context=%d  stride=%d",
            len(self.tokens), len(self._starts), context_length, stride,
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self._starts[idx]
        end   = start + self.context_length
        return (
            torch.tensor(self.tokens[start:end],    dtype=torch.long),
            torch.tensor(self.tokens[start+1:end+1], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# ProcessedDataset — loads pre-built tokenized.pt
# ---------------------------------------------------------------------------

class ProcessedDataset(Dataset):
    """
    Loads the pre-built ``tokenized.pt`` produced by ``prepare_dataset.py``.

    The file contains tensors already laid out as sliding windows, so no
    re-tokenization is required. This is the preferred dataset for training.

    Parameters
    ----------
    processed_path:
        Path to the ``tokenized.pt`` file.
    """

    def __init__(self, processed_path: Path) -> None:
        processed_path = Path(processed_path).resolve()
        if not processed_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {processed_path}\n"
                "Run:  python scripts/prepare_dataset.py"
            )

        data = torch.load(processed_path, map_location="cpu", weights_only=True)
        self.input_ids:  torch.Tensor = data["input_ids"]
        self.target_ids: torch.Tensor = data["target_ids"]
        self.vocab_size:     int = int(data["vocab_size"])
        self.context_length: int = int(data["context_length"])

        logger.info(
            "ProcessedDataset: windows=%d  context=%d  vocab=%d",
            len(self.input_ids), self.context_length, self.vocab_size,
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


# ---------------------------------------------------------------------------
# Train / validation split helper
# ---------------------------------------------------------------------------

def split_dataset(
    dataset: Dataset,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """
    Deterministically split *dataset* into train and validation subsets.

    Parameters
    ----------
    dataset:
        Full dataset to split.
    val_fraction:
        Fraction of samples reserved for validation (default: 0.05).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    train_subset, val_subset : tuple[Subset, Subset]
    """
    n = len(dataset)
    n_val   = max(1, int(n * val_fraction))
    n_train = n - n_val

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=generator,
    )
    logger.info("Split: train=%d  val=%d", n_train, n_val)
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloader(
    corpus_path: Path,
    tokenizer_path: Path,
    context_length: int,
    stride: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a DataLoader from the raw corpus (re-tokenizes every run).

    Prefer ``create_processed_dataloader`` for training runs.
    """
    dataset = TokenizerDataset(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        context_length=context_length,
        stride=stride,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_processed_dataloader(
    processed_path: Path,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    val_fraction: float = 0.0,
    seed: int = 42,
) -> DataLoader | tuple[DataLoader, DataLoader]:
    """
    Build DataLoader(s) from a pre-built ``tokenized.pt`` file.

    Parameters
    ----------
    processed_path:
        Path to ``tokenized.pt``.
    batch_size:
        Batch size.
    shuffle:
        Shuffle the training loader.
    num_workers:
        DataLoader worker processes.
    pin_memory:
        Pin memory for faster GPU transfer.
    val_fraction:
        If > 0, split and return (train_loader, val_loader).
        If 0, return a single DataLoader.
    seed:
        Random seed for the train/val split.

    Returns
    -------
    DataLoader | tuple[DataLoader, DataLoader]
    """
    dataset = ProcessedDataset(processed_path)

    if val_fraction > 0.0:
        train_sub, val_sub = split_dataset(dataset, val_fraction, seed)
        train_loader = DataLoader(
            train_sub, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_sub, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        return train_loader, val_loader

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )
