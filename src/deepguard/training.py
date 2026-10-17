"""Video-level training, fixed-encoder temporal training and genuine pair sync tasks."""
from __future__ import annotations
import copy
import json
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .data import assert_no_leakage, read_manifest
from .media import preprocess, handcrafted
from .models import make_model, load_checkpoint
from .utils import seed_all, sha256, provenance, write_json
from .evaluation import evaluate, fit_temperature

LOG = logging.getLogger(__name__)


def tensors(features: dict, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    faces = torch.as_tensor(features["faces"], device=device).float().permute(0, 3, 1, 2)/255
    audio = torch.as_tensor(features["audio_mels"], device=device).float()
    return faces, audio[:, None] if audio.ndim == 3 else audio


def sync_inputs(features: dict, device: str = "cpu", offset: int = 0) -> tuple[torch.Tensor, torch.Tensor] | None:
    mouths, audio, times = features["mouths"], features["sync_mels"], features["sync_times"]
    count = len(mouths)
    starts = [i for i in range(max(0, -offset), count - 4 - max(0, offset), 5)
              if np.diff(times[i:i+5]).max(initial=0) <= .35]
    if not starts:
        return None
    m = np.stack([mouths[i:i+5] for i in starts])
    a = np.stack([audio[i+offset:i+offset+5] for i in starts])
    return (torch.as_tensor(m, device=device).float().permute(0, 4, 1, 2, 3)/255,
            torch.as_tensor(a, device=device).float()[:, None])


def predict_logit(model: nn.Module, kind: str, features: dict, device: str,
                  encoder: nn.Module | None = None, augment: bool = False) -> torch.Tensor | None:
    faces, audio = tensors(features, device)
    if kind in {"visual", "temporal"} and len(faces) < (8 if kind == "temporal" else 1):
        return None
    if kind == "visual":
        if augment:
            if torch.rand(()) < .5:
                faces = faces.flip(-1)
            faces = (faces * (1 + (torch.rand((), device=device)-.5)*.2)).clamp(0, 1)
        # Chunk to bound CPU/GPU memory, average frame logits at video level.
        return torch.cat([model(part) for part in faces.split(8)]).mean()
    if kind == "audio":
        if not len(audio):
            return None
        if augment:
            audio = (audio + torch.randn_like(audio)*.02).clamp(0, 1)
        return model(audio).mean()
    if kind == "temporal":
        if encoder is None:
            raise ValueError("Temporal model requires the trained visual encoder")
        gaps = np.diff(features["times"])
        if gaps.max(initial=0) > .35:
            return None
        with torch.no_grad():
            embeddings = torch.cat([encoder.embed(part) for part in faces.split(8)])
        return model(embeddings[None]).mean()
    if kind == "sync":
        inputs = sync_inputs(features, device)
        return -model(*inputs).mean() if inputs else None  # mismatch logit
    raise ValueError(f"Unsupported modality: {kind}")


def label_column(kind: str) -> str:
    return "audio_label" if kind == "audio" else "visual_label" if kind in {"visual", "temporal"} else "label"


def train(manifest: str, kind: str, cfg: dict, output: str, visual_checkpoint: str | None = None,
          landmark_asset: str | None = None) -> None:
    seed_all(cfg["seed"])
    frame = read_manifest(manifest)
    if "split" not in frame:
        raise ValueError("Create identity/source splits first")
    assert_no_leakage(frame)
    device = cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if kind == "sync" and not landmark_asset:
        raise ValueError("Sync training needs a licensed FaceLandmarker asset; see DATASET_GUIDE")
    encoder = None
    if kind == "temporal":
        if not visual_checkpoint:
            raise ValueError("Train visual model before temporal training")
        encoder, meta = load_checkpoint(visual_checkpoint, device)
        if meta["kind"] != "visual":
            raise ValueError("Expected visual checkpoint")
        if meta["manifest_sha256"] != sha256(manifest):
            raise ValueError("Encoder must use the same split manifest")
    model = make_model(kind, cfg.get("pretrained", False)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    parts: dict[str, list] = {"train": [], "val": []}
    col = label_column(kind)
    for _, row in frame[frame.split.isin(parts)].iterrows():
        if kind == "sync":
            if row.visual_label != 0 or row.audio_label != 0:
                continue
        elif row[col] < 0:
            continue
        f = preprocess(row.path, cfg, "data/processed", landmark_asset)
        with torch.no_grad():
            valid = sync_inputs(f, device, 5) is not None if kind == "sync" else predict_logit(model.eval(), kind, f, device, encoder) is not None
        if valid:
            parts[row.split].append((f, int(row[col])))
    for part, rows in parts.items():
        if not rows or (kind != "sync" and len({y for _, y in rows}) < 2):
            raise ValueError(f"{part} needs usable media from both classes (real pairs for sync)")
    positive = sum(y for _, y in parts["train"])
    weight = 1. if kind == "sync" else (len(parts["train"])-positive)/positive
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    best, stale, history, state = float("inf"), 0, [], None
    start = time.perf_counter()
    for epoch in range(cfg["epochs"]):
        losses = {}
        for part, rows in parts.items():
            model.train(part == "train")
            order = np.random.permutation(len(rows)) if part == "train" else range(len(rows))
            values = []
            for index in order:
                f, y = rows[index]
                with torch.set_grad_enabled(part == "train"):
                    if kind == "sync":
                        paired = sync_inputs(f, device)
                        shifted = sync_inputs(f, device, 5)
                        a, b = model(*paired), model(*shifted)
                        logits = torch.cat([a, b])
                        target = torch.cat([torch.ones_like(a), torch.zeros_like(b)])
                    else:
                        logits = predict_logit(model, kind, f, device, encoder, part == "train" and cfg["augment"]).view(1)
                        target = logits.new_tensor([y])
                    loss = criterion(logits, target)
                    if part == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()
                    values.append(loss.item())
            losses[part] = float(np.mean(values))
        scheduler.step(losses["val"])
        history.append({"epoch": epoch+1, **losses})
        LOG.info("Epoch %s %s", epoch+1, losses)
        if losses["val"] < best:
            best, stale, state = losses["val"], 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    if state is None:
        raise ValueError("No finite validation checkpoint produced")
    metadata = {"schema": 1, "kind": kind, "state_dict": state, "config": cfg,
                "trained_samples": len(parts["train"]), "validation_samples": len(parts["val"]),
                "manifest_sha256": sha256(manifest), "provenance": provenance(),
                "temperature": 1., "calibrated": False,
                "visual_checkpoint_sha256": sha256(visual_checkpoint) if visual_checkpoint else None,
                "landmark_sha256": sha256(landmark_asset) if landmark_asset else None,
                "label_semantics": "alignment (negative logit is mismatch)" if kind == "sync" else col}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(metadata, output)
    write_json(str(output)+".history.json", {"epochs": history, "seconds": time.perf_counter()-start,
                                            "trained_samples": len(parts["train"]), "config": cfg})


def baseline(manifest: str, cfg: dict, output: str) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from xgboost import XGBClassifier
    import pickle
    seed_all(cfg["seed"])
    frame = read_manifest(manifest)
    assert_no_leakage(frame)
    x = np.stack([handcrafted(preprocess(p, cfg, "data/processed")) for p in frame.path])
    train_mask, test_mask = (frame.split == "train").to_numpy(), (frame.split == "test").to_numpy()
    if len(np.unique(frame.label[train_mask])) < 2 or not test_mask.any():
        raise ValueError("Both training classes and held-out test files required")
    models = {"logistic": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=cfg["seed"]),
              "forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=cfg["seed"], n_jobs=4),
              "xgboost": XGBClassifier(n_estimators=200, max_depth=4, random_state=cfg["seed"], n_jobs=4,
                                       scale_pos_weight=float((frame.label[train_mask]==0).sum()/max(1,(frame.label[train_mask]==1).sum())))}
    for name, classifier in models.items():
        model = make_pipeline(SimpleImputer(), StandardScaler(), classifier)
        model.fit(x[train_mask], frame.label[train_mask])
        p = model.predict_proba(x[test_mask])[:, 1]
        folder = Path(output)/name
        result = evaluate(frame.label[test_mask].to_numpy(), p, str(folder),
                          frame.loc[test_mask,"group_id"].to_numpy() if "group_id" in frame else None)
        pd.DataFrame({"path": frame.path[test_mask], "label": frame.label[test_mask], "probability": p}).to_csv(folder/"predictions.csv", index=False)
        # Local research artifact only; never accept uploaded pickle models.
        with (folder/"model.pkl").open("wb") as handle:
            pickle.dump(model, handle)
        write_json(folder/"provenance.json", {"manifest_sha256": sha256(manifest), "config": cfg, **provenance()})


def predict_manifest(manifest: str, checkpoint: str, output: str, split: str = "test",
                     visual_checkpoint: str | None = None, landmark_asset: str | None = None) -> None:
    frame = read_manifest(manifest)
    assert_no_leakage(frame)
    model, meta = load_checkpoint(checkpoint)
    seed_all(meta["config"]["seed"])
    kind = meta["kind"]
    encoder = load_checkpoint(visual_checkpoint)[0] if visual_checkpoint else None
    if kind == "temporal" and (not visual_checkpoint or sha256(visual_checkpoint) != meta["visual_checkpoint_sha256"]):
        raise ValueError("Temporal encoder checksum mismatch")
    rows = []
    for _, row in frame[frame.split == split].iterrows():
        f = preprocess(row.path, meta["config"], "data/processed", landmark_asset)
        t = time.perf_counter()
        with torch.no_grad():
            value = predict_logit(model, kind, f, "cpu", encoder)
        if value is None:
            continue
        probability = torch.sigmoid(value/meta["temperature"]).item()
        record = {"path": row.path, "label": int(row[label_column(kind)]), "overall_label": int(row.label),
                  "probability": probability, "logit": value.item(), "split": split,
                  "seconds": time.perf_counter()-t, "identities": row.identities,
                  "group_id": row.get("group_id", row.identities),
                  "quality": float(f["audio_quality"] if kind == "audio" else f["visual_quality"])}
        rows.append(record)
    if not rows:
        raise ValueError("No usable predictions executed")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    write_json(output+".meta.json", {"kind": kind, "checkpoint_sha256": sha256(checkpoint),
                                   "manifest_sha256": sha256(manifest), "split": split,
                                   "model_bytes": Path(checkpoint).stat().st_size,
                                   "coverage": len(rows)/max(1, sum(frame.split == split))})


def calibrate(checkpoint: str, predictions: str, output: str) -> None:
    table = pd.read_csv(predictions)
    if set(table.split) != {"calibration"}:
        raise ValueError("Use the reserved calibration split, not training/validation/test")
    model, meta = load_checkpoint(checkpoint)
    if meta["kind"] == "sync":
        raise ValueError("Sync alignment is not a calibrated deepfake probability")
    source = json.loads(Path(predictions+".meta.json").read_text())
    if source["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("Calibration predictions do not belong to checkpoint")
    temperature = fit_temperature(table.label.to_numpy(), table.logit.to_numpy())
    payload = torch.load(checkpoint, weights_only=True)
    payload.update(temperature=temperature, calibrated=True, calibration_sha256=sha256(predictions))
    torch.save(payload, output)
