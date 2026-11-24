"""Shared CLI plumbing: every script takes --config and repeatable --set key=value overrides."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recsys.config import Config, load_config, parse_overrides  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set data.user_fraction=0.1",
    )
    return p


def config_from(args: argparse.Namespace) -> Config:
    return load_config(args.config, parse_overrides(args.set))
