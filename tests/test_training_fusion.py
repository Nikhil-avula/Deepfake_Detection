"""Engineering-only optimizer/checkpoint/fusion integration, not benchmark data."""
import json
import numpy as np
import pandas as pd
import torch
from deepguard.training import train
from deepguard.models import load_checkpoint
from deepguard.utils import config, write_json
from deepguard.fusion import fit, assess


def test_audio_optimizer_and_safe_checkpoint_roundtrip(tmp_path,monkeypatch):
    cfg = config(); cfg.update(epochs=1,patience=1)
    rows = []
    for i in range(8):
        p = tmp_path/f"{i}.media"; p.write_bytes(str(i).encode())
        rows.append(dict(path=str(p),label=i%2,visual_label=i%2,audio_label=i%2,
                         identities=f"i{i}",sources=f"s{i}",method="engineering-only",dataset="not-research",
                         split="train" if i<4 else "val"))
    manifest = tmp_path/"input.csv"; pd.DataFrame(rows).to_csv(manifest,index=False)
    rng = np.random.default_rng(1)
    def fixture(*args,**kwargs):
        return {"faces":np.empty((0,160,160,3),dtype=np.uint8),
                "audio_mels":rng.random((1,64,80),dtype=np.float32)}
    monkeypatch.setattr("deepguard.training.preprocess",fixture)
    output = tmp_path/"audio.pt"
    train(str(manifest),"audio",cfg,str(output))
    model,meta = load_checkpoint(str(output))
    assert meta["trained_samples"] == 4
    assert model(torch.zeros(1,1,64,80)).shape == (1,)
    assert len(json.loads((tmp_path/"audio.pt.history.json").read_text())["epochs"]) == 1


def test_fusion_reserved_splits_and_artifact_evaluation(tmp_path):
    rng = np.random.default_rng(42)
    paths = {}
    for split in ["fusion","calibration","test"]:
        rows=[]
        for i in range(20):
            x = np.r_[rng.uniform(.1,.9,4),np.ones(4),np.ones(4)]
            rows.append({**{f"x{k}":v for k,v in enumerate(x)}, "label":i%2,
                         "path":f"{split}-{i}","identities":f"{split}-id{i}","split":split,
                         "simple":x[:3].mean(),"reliability":x[:3].mean()})
        p = tmp_path/f"{split}.csv"; pd.DataFrame(rows).to_csv(p,index=False)
        write_json(str(p)+".meta.json",{"checkpoint_hashes":{},"manifest_sha256":"engineering", "coverage":1.})
        paths[split] = str(p)
    checkpoint = str(tmp_path/"fusion.json")
    fit(paths["fusion"],paths["calibration"],checkpoint)
    assess(paths["test"],checkpoint,str(tmp_path/"evaluation"))
    assert (tmp_path/"evaluation/learned_score_only/metrics.json").is_file()
    assert (tmp_path/"evaluation/without_audio/predictions.csv").is_file()
