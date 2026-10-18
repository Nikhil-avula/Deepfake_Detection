"""Quality-gated video-only predictions with explicit uncertainty and dataset provenance."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from .media import sample_video,MediaError
from .models import FaceEncoder,aggregate,load_head,predict_head
from .utils import sha256


class Analyzer:
    def __init__(self,cfg: dict,model: str="checkpoints/visual.json"):
        self.cfg=cfg;self.path=Path(model)
        self.head=load_head(self.path) if self.path.exists() else None
        self.encoder=None
        if self.head:
            for key in ["frames","image_size","max_pixels","min_faces","min_face_fraction","max_multiple_fraction","min_sharpness"]:
                if cfg[key]!=self.head["config"][key]:raise ValueError("Model settings do not match video preprocessing")

    def analyze(self,path: str | Path) -> tuple[dict,dict | None]:
        started=time.perf_counter()
        report={"decision":"Cannot analyze","fake_score":None,"reasons":[],"scope":"visual facial evidence only",
                "model_status":"trained pilot" if self.head else "not trained","membership":"unknown",
                "notice":"A model score is not proof. This small pilot has not established accuracy on unfamiliar sources."}
        try:media=sample_video(path,self.cfg)
        except (MediaError,OSError) as exc:
            report["reasons"]=[str(exc)];report["seconds"]=time.perf_counter()-started
            return report,None
        report.update(metadata=media["metadata"],frames_checked=media["sampled"],faces_found=len(media["faces"]),
                      missing_faces=media["missing"],multiple_face_frames=media["multiple"])
        if not media["usable"]:report["reasons"]=media["reasons"]
        elif not self.head:report["reasons"]=["The model has not been trained yet."]
        else:
            if self.encoder is None:self.encoder=FaceEncoder()
            embeddings=self.encoder.encode(media["faces"],self.cfg["batch_size"])
            features=aggregate(embeddings,media["faces"])
            score=float(predict_head(features,self.head,calibrated=False)[0])
            z=(features-self.head["mean"])/np.asarray(self.head["scale"])
            distance=float(np.sqrt(np.mean(z**2)))
            unfamiliar=distance>self.head["ood_distance_threshold"]
            report.update(fake_score=score,feature_distance=distance,unfamiliar_features=bool(unfamiliar),
                          model_sha256=sha256(self.path),decision="Not sure")
            if unfamiliar:report["reasons"].append("This video's visual features are outside the pilot's usual range. Get an independent check.")
            elif score>=self.cfg["fake_threshold"]:report["decision"]="Likely fake"
            elif score<=self.cfg["real_threshold"]:report["decision"]="Likely real"
            else:report["reasons"].append("The result is too close to the decision boundary to give a clear answer.")
            digest=sha256(path)
            if digest in self.head.get("train_hashes",[]):
                report["membership"]="training video"
                report["reasons"].append("This exact file was used for training. Its result is not an independent test.")
            elif digest in self.head.get("test_hashes",[]):report["membership"]="held-out test video"
            report["score_type"]="uncalibrated classifier score, not probability of truth"
        report["seconds"]=time.perf_counter()-started
        return report,media
