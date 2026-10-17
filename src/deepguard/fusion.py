"""Late fusion fitted on a reserved split, separate from modality training."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .data import read_manifest, assert_no_leakage
from .inference import Analyzer, fusion_vector
from .evaluation import evaluate, fit_temperature, selective_risk
from .utils import write_json, sha256


def collect(manifest: str, cfg: dict, directory: str, split: str, output: str,
            landmark_asset: str | None = None) -> None:
    frame = read_manifest(manifest)
    assert_no_leakage(frame)
    analyzer = Analyzer(cfg, directory, landmark_asset)
    if not analyzer.models:
        raise ValueError("Train modality models before collecting fusion evidence")
    rows = []
    for _, row in frame[frame.split == split].iterrows():
        result, _ = analyzer.analyze(row.path, "data/processed")
        vector = fusion_vector(result["probabilities"], result["quality"])
        if not vector[8:].any():
            continue
        rows.append({**{f"x{i}": float(v) for i, v in enumerate(vector)},
                     "label": int(row.label), "path": row.path, "split": split,
                     "identities": row.identities, "group_id":row.get("group_id",row.identities), "simple": result["simple_fusion"],
                     "reliability": result["reliability_fusion"], "seconds": result["seconds"]})
    if not rows:
        raise ValueError("No usable fusion examples")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    write_json(output+".meta.json", {"checkpoint_hashes": analyzer.hashes,
                                   "manifest_sha256": sha256(manifest), "split": split,
                                   "coverage": len(rows)/max(1, sum(frame.split == split))})


def matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"x{i}" for i in range(12)]].to_numpy(dtype=float)


def fit(train_csv: str, calibration_csv: str, output: str) -> None:
    train, calibration = pd.read_csv(train_csv), pd.read_csv(calibration_csv)
    if set(train.split) != {"fusion"} or set(calibration.split) != {"calibration"}:
        raise ValueError("Fusion training and calibration must use their reserved splits")
    if set(train.path) & set(calibration.path):
        raise ValueError("Fusion/calibration file overlap")
    metadata = json.loads(Path(train_csv+".meta.json").read_text())
    other = json.loads(Path(calibration_csv+".meta.json").read_text())
    if metadata["checkpoint_hashes"] != other["checkpoint_hashes"] or metadata["manifest_sha256"] != other["manifest_sha256"]:
        raise ValueError("Incompatible fusion sources")
    original = matrix(train)
    augmented, targets = [original], [train.label.to_numpy()]
    # Modality dropout is training augmentation, never reported as observed evidence.
    for i in range(4):
        values = original.copy()
        values[:,i], values[:,i+4], values[:,i+8] = .5, 0., 0.
        valid = values[:,8:].sum(1) > 0
        augmented.append(values[valid]); targets.append(train.label.to_numpy()[valid])
    train_x, train_y = np.concatenate(augmented), np.concatenate(targets)
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=1., max_iter=1000, random_state=42).fit(scaler.transform(train_x), train_y)
    logits = model.decision_function(scaler.transform(matrix(calibration)))
    temperature = fit_temperature(calibration.label.to_numpy(), logits)
    write_json(output, {"schema": 1, "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
                        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                        "temperature": temperature, "calibrated": True,
                        "checkpoint_hashes": metadata["checkpoint_hashes"],
                        "manifest_sha256": metadata["manifest_sha256"],
                        "train_sha256": sha256(train_csv), "calibration_sha256": sha256(calibration_csv),
                        "trained_samples": len(train), "calibration_samples": len(calibration),
                        "optimization_rows_with_modality_dropout": len(train_x),
                        "contribution": "Quality and availability as learned late-fusion covariates"})
    # Fair learned comparator: same examples/optimizer, omit quality covariates.
    no_quality = train_x.copy()
    no_quality[:, 4:8] = 0
    plain_scaler = StandardScaler().fit(no_quality)
    plain = LogisticRegression(C=1., max_iter=1000, random_state=42).fit(plain_scaler.transform(no_quality), train_y)
    cal_plain = matrix(calibration).copy()
    cal_plain[:, 4:8] = 0
    plain_temperature = fit_temperature(calibration.label.to_numpy(), plain.decision_function(plain_scaler.transform(cal_plain)))
    write_json(output+".score_only.json", {"coef":plain.coef_[0].tolist(), "intercept":float(plain.intercept_[0]),
               "mean":plain_scaler.mean_.tolist(), "scale":plain_scaler.scale_.tolist(),
               "temperature":plain_temperature, "checkpoint_hashes":metadata["checkpoint_hashes"]})


def assess(test_csv: str, checkpoint: str, output: str) -> None:
    table = pd.read_csv(test_csv)
    if set(table.split) != {"test"}:
        raise ValueError("Use held-out test split")
    meta = json.loads(Path(test_csv+".meta.json").read_text())
    model = json.loads(Path(checkpoint).read_text())
    if meta["checkpoint_hashes"] != model["checkpoint_hashes"]:
        raise ValueError("Checkpoint provenance mismatch")
    x = matrix(table)
    y = table.label.to_numpy()
    groups = table["group_id" if "group_id" in table else "identities"].to_numpy()
    def predict(values: np.ndarray) -> np.ndarray:
        return expit((((values-model["mean"])/model["scale"]) @ np.asarray(model["coef"]) + model["intercept"])/model["temperature"])
    predictions = {"learned": predict(x)}
    plain_path = Path(checkpoint+".score_only.json")
    if plain_path.exists():
        plain = json.loads(plain_path.read_text())
        values = x.copy()
        values[:,4:8] = 0
        predictions["learned_score_only"] = expit((((values-plain["mean"])/plain["scale"]) @ np.asarray(plain["coef"])+plain["intercept"])/plain["temperature"])
    for method in ["simple", "reliability"]:
        valid = table[method].notna().to_numpy()
        if valid.any():
            evaluate(y[valid], table[method][valid].to_numpy(), str(Path(output)/method), groups[valid])
    for i, name in enumerate(["visual", "audio", "temporal", "sync"]):
        removed = x.copy()
        removed[:, i], removed[:, i+4], removed[:, i+8] = .5, 0., 0.
        predictions[f"without_{name}"] = predict(removed)
    for name, p in predictions.items():
        evaluate(y, p, str(Path(output)/name), groups)
        pd.DataFrame({"path": table.path, "label": y, "probability": p}).to_csv(Path(output)/name/"predictions.csv", index=False)
        write_json(Path(output)/name/"selective_risk.json", selective_risk(y,p))
    write_json(Path(output)/"coverage.json", {"coverage": meta["coverage"], "n": len(table),
               "note": "Ablations remove evidence at inference; retrained ablations remain separate experiments."})
