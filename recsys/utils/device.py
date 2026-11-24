"""Device selection: CUDA if present, else Apple MPS, else CPU. Overridable via config/env."""

from __future__ import annotations

import os

import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device(preferred: str | None = None) -> torch.device:
    """Pick a torch device.

    Order: explicit `preferred` (or RECSYS_DEVICE env var) > cuda > mps > cpu.
    "auto" or None means pick automatically. Always use float32 tensors (MPS has no float64).
    """
    choice = (preferred or os.environ.get("RECSYS_DEVICE") or "auto").lower()
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
