"""Run metadata for every results/*.json: seed, git commit, hardware, library versions."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

_LIBS = [
    "torch",
    "polars",
    "numpy",
    "faiss-cpu",
    "lightgbm",
    "implicit",
    "onnxruntime",
    "scikit-learn",
    "sentence-transformers",
    "fastapi",
    "redis",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        # Outputs (results/, docs/) don't count as "dirty"; code and configs do.
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD", "--", ".", ":!results", ":!docs"],
            stderr=subprocess.DEVNULL,
        )
        return sha.decode().strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def hardware() -> dict[str, Any]:
    """Describe the machine the numbers came from (CPU model, cores, RAM, GPU)."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        if platform.system() == "Darwin":
            try:
                cpu = (
                    subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
                    .decode()
                    .strip()
                )
            except Exception:
                pass
    info["cpu"] = cpu
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["mps"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except ImportError:  # pragma: no cover
        pass
    return info


def hardware_label() -> str:
    h = hardware()
    accel = h.get("cuda") or ("Apple MPS" if h.get("mps") else "no GPU")
    return f"{h['cpu']} ({h['cpu_count']} vCPU, {h.get('ram_gb', '?')} GB RAM, {accel})"


def lib_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for lib in _LIBS:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = "not installed"
    return out


def run_metadata(cfg: Any, **extra: Any) -> dict[str, Any]:
    d = cfg.data
    return {
        "dataset": "MovieLens-32M (ml-32m.zip, md5 d472be332d4daa821edc399621853b57)"
        if "ml-32m" in d.raw_dir
        else d.raw_dir,
        "split": {
            "val_start": d.val_start,
            "test_start": d.test_start,
            "test_end": d.test_end,
            "positive_threshold": d.positive_threshold,
            "user_fraction": d.user_fraction,
        },
        "seed": cfg.seed,
        "config": cfg.path,
        "git_commit": git_commit(),
        "hardware": hardware(),
        "libraries": lib_versions(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }


def write_result(path: str | Path, payload: dict[str, Any], cfg: Any, **extra: Any) -> Path:
    """Write results JSON with run metadata attached under "meta"."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"meta": run_metadata(cfg, **extra), **payload}
    path.write_text(json.dumps(out, indent=2, default=_json_default))
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


class PeakMemory:
    """Context manager that samples process RSS in a thread and records the peak (GB)."""

    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.peak_gb = 0.0
        self.seconds = 0.0

    def __enter__(self) -> PeakMemory:
        import threading

        import psutil

        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._t0 = time.perf_counter()

        def _loop() -> None:
            while not self._stop.is_set():
                self.peak_gb = max(self.peak_gb, self._proc.memory_info().rss / 2**30)
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_gb = max(self.peak_gb, self._proc.memory_info().rss / 2**30)
        self.seconds = time.perf_counter() - self._t0
