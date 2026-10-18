"""Held-out video-level metrics, group bootstrap and reliability diagrams."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn import metrics
from scipy.optimize import minimize_scalar
from .utils import write_json


def score(y,p) -> dict:
    y,p=np.asarray(y),np.asarray(p,dtype=float)
    if not len(y) or y.shape!=p.shape or not set(y)<={0,1}:raise ValueError("Nonempty matching binary targets required")
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError("Invalid probabilities")
    pred=p>=.5
    tn,fp,fn,tp=metrics.confusion_matrix(y,pred,labels=[0,1]).ravel()
    ece=0.;bins=np.minimum((p*10).astype(int),9)
    for i in range(10):
        mask=bins==i
        if mask.any():ece+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
    both=len(np.unique(y))==2
    precision,recall,_=metrics.precision_recall_curve(y,p) if both else (None,None,None)
    return {"videos":len(y),"accuracy":float(metrics.accuracy_score(y,pred)),
        "balanced_accuracy":float(metrics.balanced_accuracy_score(y,pred)),
        "precision":float(metrics.precision_score(y,pred,zero_division=0)),
        "recall":float(metrics.recall_score(y,pred,zero_division=0)),
        "f1":float(metrics.f1_score(y,pred,zero_division=0)),
        "roc_auc":float(metrics.roc_auc_score(y,p)) if both else None,
        "pr_auc":float(metrics.auc(recall,precision)) if both else None,
        "average_precision":float(metrics.average_precision_score(y,p)) if both else None,
        "confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]],
        "false_positive_rate":float(fp/(fp+tn)) if fp+tn else None,
        "false_negative_rate":float(fn/(fn+tp)) if fn+tp else None,
        "brier":float(metrics.brier_score_loss(y,p)),"ece":float(ece)}


def temperature(y,logits) -> float:
    y=np.asarray(y);logits=np.asarray(logits)
    if len(y)<4 or set(y)!={0,1}:raise ValueError("Separate calibration examples from both classes required")
    return float(minimize_scalar(lambda t:np.mean(np.logaddexp(0,logits/t)-y*logits/t),bounds=(.05,20),method="bounded").x)


def evaluate(y,p,groups,output: str) -> dict:
    y,p,groups=np.asarray(y),np.asarray(p),np.asarray(groups)
    result=score(y,p)
    unique=np.unique(groups);rng=np.random.default_rng(42);accuracy=[]
    for _ in range(1000):
        ids=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique))])
        accuracy.append(float(np.mean((p[ids]>=.5)==y[ids])))
    result["source_groups"]=len(unique)
    result["accuracy_group_bootstrap_95_ci"]=np.quantile(accuracy,[.025,.975]).tolist()
    result["interval_limit"]="Source-pair bootstrap; identity independence is unverified; small-sample intervals are unstable."
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/"metrics.json",result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import CalibrationDisplay
    fig,axes=plt.subplots(1,4,figsize=(16,3.5))
    metrics.ConfusionMatrixDisplay.from_predictions(y,p>=.5,labels=[0,1],display_labels=["Real","Fake"],ax=axes[0],colorbar=False)
    if len(np.unique(y))==2:
        metrics.RocCurveDisplay.from_predictions(y,p,ax=axes[1])
        metrics.PrecisionRecallDisplay.from_predictions(y,p,ax=axes[2])
    CalibrationDisplay.from_predictions(y,p,n_bins=5,ax=axes[3])
    fig.tight_layout();fig.savefig(folder/"evaluation.png",dpi=150);plt.close(fig)
    risk=[]
    for band in [0,.05,.1,.15,.2,.3,.4]:
        keep=np.abs(p-.5)>=band
        risk.append({"band":band,"coverage":float(keep.mean()),"error_rate":float(np.mean((p[keep]>=.5)!=y[keep])) if keep.any() else None})
    write_json(folder/"selective_risk.json",risk)
    return result
