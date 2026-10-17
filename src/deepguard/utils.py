"""Configuration, provenance and reproducibility helpers."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml


def config(path: str | Path = "configs/default.yaml") -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


def provenance() -> dict:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "cuda": torch.cuda.is_available(),
            "cpus": os.cpu_count(), "git_commit": result.stdout.strip(),
            "git_dirty": bool(dirty.stdout.strip())}
