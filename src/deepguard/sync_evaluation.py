"""Held-out genuine-pair evaluation across explicitly declared audio offsets."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import read_manifest, assert_no_leakage
from .media import preprocess
from .models import load_checkpoint
from .training import sync_inputs
from .evaluation import evaluate
from .utils import write_json, sha256


def assess_sync(manifest: str, checkpoint: str, asset: str, output: str) -> None:
    frame = read_manifest(manifest)
    assert_no_leakage(frame)
    model, meta = load_checkpoint(checkpoint)
    if meta["kind"] != "sync" or sha256(asset) != meta["landmark_sha256"]:
        raise ValueError("Sync model/landmark asset provenance mismatch")
    records = []
    for _, row in frame[(frame.split == "test") & (frame.visual_label == 0) & (frame.audio_label == 0)].iterrows():
        features = preprocess(row.path, meta["config"], "data/processed", asset)
        for offset in [-10,-5,0,5,10]:
            inputs = sync_inputs(features, offset=offset)
            if inputs is None:
                continue
            with torch.no_grad():
                matched = torch.sigmoid(model(*inputs)).mean().item()
            records.append({"path":row.path,"identities":row.identities,"group_id":row.get("group_id",row.identities),"offset_samples":offset,
                            "nominal_offset_seconds":offset/meta["config"]["sample_fps"],
                            "label":int(offset != 0),"probability":1-matched})
    if not records or len({r["label"] for r in records}) < 2:
        raise ValueError("Need usable genuine held-out recordings and nonzero offsets")
    table = pd.DataFrame(records)
    Path(output).mkdir(parents=True,exist_ok=True)
    table.to_csv(Path(output)/"pair_predictions.csv",index=False)
    evaluate(table.label.to_numpy(),table.probability.to_numpy(),output,table.group_id.to_numpy())
    write_json(Path(output)/"task.json", {"task":"Within-recording aligned vs time-shifted genuine pairs, not deepfake classification",
               "checkpoint_sha256":sha256(checkpoint),"manifest_sha256":sha256(manifest),
               "natural_delay_controls":"Not supplied; still required before sync validity claims"})
