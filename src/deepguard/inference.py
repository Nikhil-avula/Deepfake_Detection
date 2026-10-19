"""Quality-gated video-only predictions with explicit uncertainty and dataset provenance."""
from __future__ import annotations
import time
from http.client import HTTPException
from pathlib import Path
import numpy as np
from .media import sample_video,MediaError,SetupError
from .models import FaceEncoder,aggregate,load_head,predict_head
from .utils import sha256,ROOT


class Analyzer:
    def __init__(self,cfg: dict,model: str="checkpoints/visual.json"):
        self.cfg=cfg;self.path=Path(model)
        self.setup_problem=None
        try:self.head=load_head(self.path) if self.path.exists() else None
        except (OSError,ValueError,KeyError,TypeError):
            self.head=None;self.setup_problem='The trained model file is damaged or incompatible. Restore checkpoints/visual.json from the project checkout.'
        self.encoder=None
        self.retry=None
        if self.head:
            settings=self.head.get('config',{})
            if cfg.get('face_detector','haar')!=settings.get('face_detector','haar'):
                self.setup_problem='The model and face detector settings do not match. Restore the model and configuration from the same training run.'
            for key in ["frames","image_size","max_pixels","min_faces","min_face_fraction","max_multiple_fraction","min_sharpness","dominant_face_area_ratio","video_sampler"]:
                if cfg.get(key)!=settings.get(key):self.setup_problem='Model settings do not match video preprocessing. Restore the model and configuration together.'
            threshold=self.head.get('ood_distance_threshold')
            if not isinstance(threshold,(int,float)) or not np.isfinite(threshold) or threshold<=0:
                self.setup_problem='The model is missing valid uncertainty settings. Restore the trained model from the project checkout.'

    def analyze(self,path: str | Path) -> tuple[dict,dict | None]:
        started=time.perf_counter()
        result,media=self._analyze(path)
        result['analysis_pass']='standard'
        retry_path=self.cfg.get('retry_model')
        if result['decision']=='Cannot analyze' and self.head and retry_path:
            if self.retry is None:
                retry_path=Path(retry_path)
                if not retry_path.is_absolute():retry_path=ROOT/retry_path
                try:
                    head=load_head(retry_path)
                    retry_cfg=dict(head['config']);retry_cfg.pop('retry_model',None)
                    self.retry=Analyzer(retry_cfg,str(retry_path))
                except (OSError,ValueError,KeyError,TypeError):
                    result.update(decision='Setup required',reasons=['The retry model is missing or damaged. Restore checkpoints/coverage_retry.json from the updated project.'])
                    result['seconds']=time.perf_counter()-started
                    return result,media
            first=result
            result,media=self.retry.analyze(path)
            result['analysis_pass']='additional face search'
            result['first_pass_faces']=first.get('faces_found',0)
            if result.get('fake_score') is not None:
                result['reasons'].insert(0,'The first face search was insufficient. This result uses additional samples and a separately trained classifier for the foreground-face pipeline.')
        result['seconds']=time.perf_counter()-started
        return result,media

    def _analyze(self,path: str | Path) -> tuple[dict,dict | None]:
        started=time.perf_counter()
        report={"decision":"Cannot analyze","fake_score":None,"reasons":[],"scope":"visual facial evidence only",
                "model_status":"trained pilot" if self.head else "not trained","membership":"unknown",
                "notice":"A model score is not proof. This small pilot has not established accuracy on unfamiliar sources."}
        if self.setup_problem:
            report.update(decision='Setup required',reasons=[self.setup_problem],seconds=time.perf_counter()-started)
            return report,None
        try:media=sample_video(path,self.cfg)
        except SetupError as exc:
            report["decision"]="Setup required";report["reasons"]=[str(exc)]
            report["seconds"]=time.perf_counter()-started
            return report,None
        except (MediaError,OSError) as exc:
            report["reasons"]=[str(exc)];report["seconds"]=time.perf_counter()-started
            return report,None
        report.update(metadata=media["metadata"],frames_checked=media["sampled"],faces_found=len(media["faces"]),
                      missing_faces=media["missing"],multiple_face_frames=media["multiple"])
        report['dominant_face_frames']=media.get('dominant_face_frames',0)
        if report['dominant_face_frames']:
            report['reasons'].append('We checked the largest foreground face. Smaller background faces were not classified.')
        if not self.head:
            report["decision"]="Setup required"
            report["reasons"]=["The trained model file is missing: checkpoints/visual.json. Update your checkout with git pull or restore the trained model file, then restart the application."]
        elif not media["usable"]:report["reasons"]=media["reasons"]
        else:
            if self.encoder is None:
                try:self.encoder=FaceEncoder()
                except (OSError,RuntimeError,ValueError,HTTPException):
                    report.update(decision='Setup required',reasons=['The image encoder could not load. Its first download needs internet access; restore the verified pretrained weights if the cache is damaged.'],seconds=time.perf_counter()-started)
                    return report,media
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
            elif digest in self.head.get('validation_hashes',[]):
                report['membership']='validation video';report['reasons'].append('This file helped choose model settings; it is not an independent test.')
            elif digest in self.head.get('calibration_hashes',[]):
                report['membership']='calibration video';report['reasons'].append('This file was used to adjust scores; use a reserved or external video for an independent check.')
            report["score_type"]="uncalibrated classifier score, not probability of truth"
        report["seconds"]=time.perf_counter()-started
        return report,media
