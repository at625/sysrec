from __future__ import annotations

import torch

from recsys.utils.device import get_device
from recsys.utils.repro import hardware, lib_versions


def test_get_device_override() -> None:
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_auto_returns_available() -> None:
    dev = get_device("auto")
    assert dev.type in {"cpu", "cuda", "mps"}


def test_metadata_fields() -> None:
    assert "cpu" in hardware()
    assert "python" in lib_versions()
