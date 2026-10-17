"""Explicit labels and connected-component splits prevent identity/source leakage."""
from __future__ import annotations
from pathlib import Path
import json
import urllib.request
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from .utils import sha256, write_json

REQUIRED = {"path", "label", "visual_label", "audio_label", "identities", "sources", "method", "dataset"}


def read_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if not len(frame) or frame.path.duplicated().any():
        raise ValueError("Manifest must be nonempty with unique paths")
    for key in ["label", "visual_label", "audio_label"]:
        frame[key] = pd.to_numeric(frame[key], errors="raise").astype(int)
        allowed = {0, 1} if key == "label" else {-1, 0, 1}
        if not set(frame[key]) <= allowed:
            raise ValueError(f"Invalid {key}; -1 means unavailable only for modality labels")
    for key in ["identities", "sources", "dataset", "method"]:
        if frame[key].str.strip().eq("").any():
            raise ValueError(f"Missing {key}; do not guess identities or provenance")
    for p in frame.path:
        if not Path(p).is_file():
            raise ValueError(f"Media file missing: {p}")
    if ((frame.label == 0) & ((frame.visual_label == 1) | (frame.audio_label == 1))).any():
        raise ValueError("Overall real label conflicts with a known manipulated modality")
    return frame


def entities(row: pd.Series) -> set[str]:
    # All donor/target identities and all parent recordings must be listed.
    return {f"{col}:{token.strip()}" for col in ["identities", "sources"]
            for token in str(row[col]).split("|") if token.strip()}


def assert_no_leakage(frame: pd.DataFrame) -> None:
    owner: dict[str, str] = {}
    for _, row in frame.iterrows():
        for key in entities(row) | {"path:" + str(Path(row.path).resolve())}:
            if key in owner and owner[key] != row.split:
                raise ValueError(f"Split leakage: {key}")
            owner[key] = row.split
    if "sha256" in frame:
        if frame.groupby("sha256").split.nunique().gt(1).any():
            raise ValueError("Duplicate content crosses splits")


def split_manifest(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Group every connected identity/source, even when derivatives link groups."""
    frame = frame.reset_index(drop=True).copy()
    parent = list(range(len(frame)))
    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen: dict[str, int] = {}
    for i, row in frame.iterrows():
        keys = entities(row)
        if "sha256" in frame:
            keys.add("hash:" + str(row.sha256))
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    groups: dict[int, list[int]] = {}
    for i in range(len(frame)):
        groups.setdefault(root(i), []).append(i)
    if len(groups) < 5:
        raise ValueError("Need at least five disconnected components for train/val/fusion/calibration/test")
    order = list(groups)
    np.random.default_rng(seed).shuffle(order)
    # Keep components intact; exact class balance is subordinate to leakage safety.
    n = len(order)
    ntrain = min(n - 4, max(1, round(n * .6)))
    nval = min(n - ntrain - 3, max(1, round(n * .1)))
    nfusion = min(n - ntrain - nval - 2, max(1, round(n * .1)))
    ncal = min(n - ntrain - nval - nfusion - 1, max(1, round(n * .1)))
    frame["split"] = "test"
    frame["group_id"] = ""
    for index, group in enumerate(order):
        name = ("train" if index < ntrain else "val" if index < ntrain+nval else
                "fusion" if index < ntrain+nval+nfusion else
                "calibration" if index < ntrain+nval+nfusion+ncal else "test")
        frame.loc[groups[group], "split"] = name
        frame.loc[groups[group], "group_id"] = f"component-{group}"
    assert_no_leakage(frame)
    return frame


def inventory(manifest: str, output: str) -> None:
    from .media import probe
    frame = read_manifest(manifest)
    records = []
    for _, row in frame.iterrows():
        record = row.to_dict()
        try:
            record.update(probe(Path(row.path)))
            record["sha256"] = sha256(row.path)
            record["error"] = ""
        except (ValueError, RuntimeError, OSError) as exc:
            record["error"] = str(exc)
        records.append(record)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    write_json(Path(output).with_suffix(".summary.json"), {
        "samples": len(frame), "datasets": frame.dataset.value_counts().to_dict(),
        "labels": frame.label.value_counts().to_dict(),
        "methods": frame.method.value_counts().to_dict(),
        "corrupt": sum(bool(r["error"]) for r in records), "manifest_sha256": sha256(manifest)})


def download(url: str, target: str, expected_sha256: str) -> None:
    """Resume HTTPS downloads only with a required, independently supplied checksum.

    Caller must have permission and verify the official origin. Never executes files.
    """
    if urlparse(url).scheme != "https" or len(expected_sha256) != 64:
        raise ValueError("HTTPS URL and official SHA-256 required")
    dest = Path(target)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(req, timeout=60) as response:
        if urlparse(response.url).scheme != "https":
            raise ValueError("Refusing non-HTTPS redirect")
        resume = offset and response.status == 206
        if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Incorrect server resume range")
        with partial.open("ab" if resume else "wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    if sha256(partial) != expected_sha256.lower():
        raise ValueError("Checksum mismatch; partial retained for diagnosis, not used")
    partial.replace(dest)
