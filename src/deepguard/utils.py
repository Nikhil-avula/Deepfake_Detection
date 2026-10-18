"""Reproducible configuration, hashes and JSON artifacts."""
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

ROOT = Path(__file__).resolve().parents[2]


def config(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or ROOT/"configs/default.yaml").read_text())


def seed_all(seed: int = 42, threads: int = 4) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(min(threads,os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True,warn_only=True)


def sha256(path: str | Path) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024**2),b""):h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def provenance() -> dict:
    result=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True)
    return {"python":platform.python_version(),"platform":platform.platform(),
            "torch":str(torch.__version__),"cuda":torch.cuda.is_available(),
            "commit":result.stdout.strip(),"dirty":bool(subprocess.run(["git","status","--porcelain"],cwd=ROOT,capture_output=True,text=True).stdout)}
