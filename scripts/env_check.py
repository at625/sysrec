"""Environment check: OS, disk, RAM, Python/torch, accelerator sanity, key imports, Docker.

Exits non-zero if a required check fails. Docker is reported but optional (only M5 compose needs it).
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED_DISK_GB = 30


def os_version() -> str:
    if platform.system() == "Darwin":
        try:
            return (
                "macOS " + subprocess.check_output(["sw_vers", "-productVersion"]).decode().strip()
            )
        except Exception:
            pass
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip('"') + f" (kernel {platform.release()})"
    except OSError:
        pass
    return platform.platform()


def accel_check() -> tuple[bool, str]:
    import torch

    from recsys.utils.device import get_device

    dev = get_device()
    a = torch.randn(512, 512, dtype=torch.float32)
    b = torch.randn(512, 512, dtype=torch.float32)
    ref = a @ b
    if dev.type == "cpu":
        return True, "device=cpu (no CUDA/MPS available); CPU matmul OK"
    t0 = time.perf_counter()
    out = (a.to(dev) @ b.to(dev)).cpu()
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    return (
        ok,
        f"device={dev}; matmul vs CPU max abs diff={float((out - ref).abs().max()):.2e} ({time.perf_counter() - t0:.3f}s)",
    )


def main() -> int:
    import psutil

    from recsys.utils.repro import hardware

    ok = True
    hw = hardware()
    print(f"OS:      {os_version()}")
    print(f"CPU:     {hw['cpu']} x{hw['cpu_count']}")
    print(
        f"RAM:     {hw.get('ram_gb')} GB total, {psutil.virtual_memory().available / 2**30:.1f} GB available"
    )
    free = shutil.disk_usage(ROOT).free / 2**30
    disk_ok = free >= REQUIRED_DISK_GB
    print(f"Disk:    {free:.1f} GB free {'OK' if disk_ok else f'WARNING: < {REQUIRED_DISK_GB} GB'}")
    print(f"Python:  {platform.python_version()}")
    import torch

    print(
        f"torch:   {torch.__version__} (cuda={torch.cuda.is_available()}, "
        f"mps={bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())})"
    )
    a_ok, msg = accel_check()
    print(f"Accel:   {'OK' if a_ok else 'FAIL'} {msg}")
    ok &= a_ok
    for mod in [
        "faiss",
        "lightgbm",
        "polars",
        "onnxruntime",
        "implicit",
        "sentence_transformers",
        "fastapi",
        "redis",
    ]:
        try:
            m = importlib.import_module(mod)
            print(f"import {mod:22s} OK {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            print(f"import {mod:22s} FAIL {e}")
            ok = False
    docker = shutil.which("docker")
    if docker:
        r = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True
        )
        running = r.returncode == 0 and r.stdout.strip()
        print(
            f"Docker:  CLI found; daemon {'running ' + r.stdout.strip() if running else 'NOT running'}"
            " (needed only for M5 docker compose)"
        )
    else:
        print("Docker:  not installed (needed only for M5 docker compose)")
    print("ENV CHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
