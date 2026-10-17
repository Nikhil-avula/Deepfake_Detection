"""Quality-gated evidence and abstention. No checkpoint means no predictions."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from .media import preprocess
from .models import load_checkpoint, reliability_fusion, disagreement
from .training import predict_logit
from .utils import sha256

MODALITIES = ("visual", "audio", "temporal", "sync")


def fusion_vector(probabilities: dict, qualities: dict) -> np.ndarray:
    return np.asarray([probabilities.get(k, .5) for k in MODALITIES] +
                      [qualities.get(k, 0.) for k in MODALITIES] +
                      [float(k in probabilities) for k in MODALITIES], dtype=np.float32)


class Analyzer:
    def __init__(self, cfg: dict, directory: str = "checkpoints", landmark_asset: str | None = None):
        self.cfg, self.directory, self.landmark_asset = cfg, Path(directory), landmark_asset
        self.models, self.meta, self.hashes = {}, {}, {}
        for kind in MODALITIES:
            path = self.directory / f"{kind}.pt"
            if path.exists():
                model, meta = load_checkpoint(str(path))
                if meta["kind"] != kind:
                    raise ValueError(f"Wrong checkpoint type: {kind}")
                # Feature extraction settings must match training settings.
                keys = ["sample_fps", "max_frames", "image_size", "sample_rate", "n_mels", "audio_seconds"]
                if any(cfg[k] != meta["config"][k] for k in keys):
                    raise ValueError("Inference feature configuration differs from training")
                self.models[kind], self.meta[kind], self.hashes[kind] = model, meta, sha256(path)
        if "temporal" in self.meta:
            if self.meta["temporal"]["visual_checkpoint_sha256"] != self.hashes.get("visual"):
                raise ValueError("Temporal visual encoder checksum mismatch")
        self.fusion = None
        path = self.directory / "fusion.json"
        if path.exists():
            self.fusion = json.loads(path.read_text())
            if self.fusion["checkpoint_hashes"] != self.hashes:
                raise ValueError("Fusion provenance differs from installed modality checkpoints")

    def analyze(self, path: str, cache: str | None = None, omit: str | None = None) -> tuple[dict, dict]:
        start = time.perf_counter()
        f = preprocess(path, self.cfg, cache, self.landmark_asset)
        p, q, reasons = {}, {}, []
        multiple = int(f["multiple_faces"]) > 0
        for kind, model in self.models.items():
            if kind == omit:
                continue
            quality = float(f["audio_quality"] if kind == "audio" else f["visual_quality"])
            if kind in {"visual", "temporal", "sync"} and (
                    multiple or float(f["face_fraction"]) < self.cfg["min_face_fraction"] or len(f["faces"]) < self.cfg["min_frames"]):
                reasons.append(f"{kind}: ambiguous or insufficient face evidence")
                continue
            if kind == "sync":
                quality = min(quality, float(f["audio_quality"]))
                if self.meta[kind].get("landmark_sha256") != (sha256(self.landmark_asset) if self.landmark_asset else None):
                    reasons.append("sync: landmark asset missing or changed")
                    continue
            if quality < self.cfg["min_quality"]:
                reasons.append(f"{kind}: below quality gate")
                continue
            with torch.no_grad():
                value = predict_logit(model, kind, f, "cpu", self.models.get("visual"))
            if value is not None:
                p[kind] = torch.sigmoid(value/self.meta[kind]["temperature"]).item()
                q[kind] = quality
        # Alignment mismatch is not a fake probability: exclude it from direct averaging.
        detection_p = {k: v for k, v in p.items() if k != "sync"}
        simple = float(np.mean(list(detection_p.values()))) if detection_p else None
        reliability = None
        if detection_p:
            reliability = reliability_fusion(torch.tensor(list(detection_p.values())),
                                             torch.tensor([q[k] for k in detection_p]),
                                             torch.ones(len(detection_p))).item()
        learned = None
        if self.fusion and p:
            vector = fusion_vector(p, q)
            raw = float(((vector - self.fusion["mean"]) / self.fusion["scale"]) @ np.asarray(self.fusion["coef"]) + self.fusion["intercept"])
            learned = float(expit(raw/self.fusion.get("temperature", 1.)))
        # Final decisions require a separately calibrated learned fusion model.
        status, confidence = "Insufficient Evidence", None
        if not self.models:
            reasons.append("Model not trained")
        elif p:
            status = "Inconclusive"
            if learned is not None and self.fusion.get("calibrated", False):
                if learned >= self.cfg["decision_high"]:
                    status, confidence = "Deepfake", learned
                elif learned <= self.cfg["decision_low"]:
                    status, confidence = "Real", 1-learned
            else:
                reasons.append("No separately calibrated fusion model; evidence scores are uncalibrated")
        disagreement_value = disagreement(p)
        if disagreement_value is not None and disagreement_value > .5:
            status, confidence = "Inconclusive", None
            reasons.append("Visual and audio evidence strongly disagree")
        report = {"decision": status, "calibrated_confidence": confidence,
                  "probabilities": p, "quality": q, "simple_fusion": simple,
                  "reliability_fusion": reliability, "learned_fusion": learned,
                  "disagreement": disagreement_value, "reasons": reasons,
                  "checkpoint_hashes": self.hashes, "metadata": json.loads(str(f["metadata"])),
                  "seconds": time.perf_counter()-start,
                  "caution": "Model evidence is not proof. Sync mismatch can reflect dubbing or recording delay.",
                  "sync_semantics": "Mismatch score, not a standalone deepfake probability"}
        return report, f
