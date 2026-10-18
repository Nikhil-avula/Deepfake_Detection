"""Frozen ImageNet EfficientNet-B0 features and a genuinely fitted visual head."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
from .media import texture_features
from .utils import sha256,write_json


class FaceEncoder(nn.Module):
    def __init__(self,pretrained: bool=True):
        super().__init__()
        self.network=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        if pretrained:
            asset=Path(torch.hub.get_dir())/"checkpoints/efficientnet_b0_rwightman-7f5810bc.pth"
            if sha256(asset)!="7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934":
                raise ValueError("The cached official backbone failed integrity verification.")
        self.network.classifier=nn.Identity()
        self.eval()

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
        std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        return self.network((x-mean)/std)

    def encode(self,faces: np.ndarray,batch: int=8) -> np.ndarray:
        if len(faces)==0:raise ValueError("No face crops available")
        x=torch.from_numpy(faces).permute(0,3,1,2).float()/255
        with torch.inference_mode():return torch.cat([self(p) for p in x.split(batch)]).numpy()


def aggregate(embeddings: np.ndarray,faces: np.ndarray) -> np.ndarray:
    return np.r_[embeddings.mean(0),embeddings.std(0),texture_features(faces)].astype(np.float32)


def cache_key(path: str | Path,cfg: dict) -> str:
    feature_cfg={k:cfg[k] for k in ["frames","image_size","max_duration","max_upload_mb","max_pixels","min_faces","min_face_fraction","max_multiple_fraction","min_sharpness"]}
    return sha256(path)+"-"+hashlib.sha256(json.dumps(feature_cfg,sort_keys=True).encode()).hexdigest()[:16]


def predict_head(features: np.ndarray,head: dict,calibrated: bool=True) -> np.ndarray:
    from scipy.special import expit
    x=np.atleast_2d(features)
    logits=((x-np.asarray(head["mean"]))/np.asarray(head["scale"])) @ np.asarray(head["coef"])+head["intercept"]
    return expit(logits/(head.get("temperature",1.) if calibrated else 1.))


def load_head(path: str | Path) -> dict:
    head=json.loads(Path(path).read_text())
    if head.get("schema")!=2 or head.get("kind")!="visual" or head.get("training_videos",0)<1:
        raise ValueError("This is not a trained video-only model artifact.")
    size=len(head.get("coef",[]))
    arrays=[np.asarray(head[k],dtype=float) for k in ["mean","scale","coef"]]
    if size!=2568 or any(a.shape!=(size,) or not np.isfinite(a).all() for a in arrays) or (arrays[1]<=0).any():
        raise ValueError("Invalid visual classifier parameters.")
    if not np.isfinite(head["intercept"]) or not .05<=head.get("temperature",1.)<=20:
        raise ValueError("Invalid model calibration parameters.")
    return head
