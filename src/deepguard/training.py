"""Train a small-data visual classifier with independent validation/calibration/test sets."""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .media import sample_video
from .models import FaceEncoder,aggregate,cache_key,predict_head
from .data import assert_no_leakage
from .evaluation import score,evaluate,temperature
from .utils import seed_all,sha256,write_json,provenance

LOG=logging.getLogger(__name__)


def prepare_features(manifest: str,cfg: dict,cache: str="data/processed") -> pd.DataFrame:
    seed_all(cfg["seed"],cfg["threads"])
    frame=pd.read_csv(manifest,keep_default_na=False,dtype={"source_group":str})
    assert_no_leakage(frame)
    encoder=None; rows=[];folder=Path(cache);folder.mkdir(parents=True,exist_ok=True)
    for i,(_,row) in enumerate(frame.iterrows()):
        record=row.to_dict();path=folder/(cache_key(row.path,cfg)+".npz")
        started=time.perf_counter()
        if path.exists():
            with np.load(path,allow_pickle=False) as f:
                usable=bool(f["usable"]);reason=str(f["reason"]);count=len(f["faces"])
        else:
            media=sample_video(row.path,cfg)
            usable=media["usable"];reason=" ".join(media["reasons"]);count=len(media["faces"])
            if count:
                if encoder is None:encoder=FaceEncoder()
                embeddings=encoder.encode(media["faces"],cfg["batch_size"])
                features=aggregate(embeddings,media["faces"])
            else:embeddings=np.empty((0,1280));features=np.zeros(2568,dtype=np.float32)
            np.savez_compressed(path,features=features,embeddings=embeddings,faces=media["faces"],times=media["times"],
                sharpness=media["sharpness"],sizes=media["sizes"],usable=np.asarray(usable),reason=np.asarray(reason))
        record.update(feature_path=str(path.resolve()),usable=usable,face_crops=count,quality_reason=reason,
                      preprocessing_seconds=time.perf_counter()-started)
        rows.append(record)
        LOG.info("Features %d/%d %s: %d face crops, usable=%s",i+1,len(frame),row.file,count,usable)
    output=pd.DataFrame(rows)
    output.to_csv(Path(manifest).with_name("prepared.csv"),index=False)
    return output


def train(manifest: str,cfg: dict,output: str="checkpoints/visual.json",results: str="reports/results") -> dict:
    started=time.perf_counter();seed_all(cfg["seed"],cfg["threads"])
    frame=prepare_features(manifest,cfg)
    usable=frame[frame.usable].copy().reset_index(drop=True)
    assert_no_leakage(usable)
    for name in ["train","val","calibration","test"]:
        if set(usable.loc[usable.split==name,"label"])!={0,1}:raise ValueError(f"Usable {name} videos must include both classes")
    x=np.stack([np.load(p,allow_pickle=False)["features"] for p in usable.feature_path])
    y=usable.label.to_numpy()
    masks={name:(usable.split==name).to_numpy() for name in ["train","val","calibration","test"]}
    folder=Path(results);folder.mkdir(parents=True,exist_ok=True)
    choices=[];fitted=[]
    for c in [.001,.01,.1,1.,10.]:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight="balanced",max_iter=2000,random_state=cfg["seed"]))
        model.fit(x[masks["train"]],y[masks["train"]])
        result=score(y[masks["val"]],model.predict_proba(x[masks["val"]])[:,1])
        choices.append({"C":c,"validation":result});fitted.append(model)
    index=max(range(len(choices)),key=lambda i:(choices[i]["validation"]["roc_auc"],-choices[i]["validation"]["brier"]))
    model=fitted[index];scaler=model[0];clf=model[1]
    calibration_logits=model.decision_function(x[masks["calibration"]])
    fitted_temperature=temperature(y[masks["calibration"]],calibration_logits)
    z=(x-scaler.mean_)/scaler.scale_
    reference=masks["train"]|masks["val"]
    distance=np.sqrt(np.mean(z**2,axis=1))
    threshold=float(np.quantile(distance[reference],.99)*1.5)
    head={"schema":2,"kind":"visual","problem_statement":"Classify facial visual evidence in an uploaded video as real or manipulated, with uncertainty and limitations.",
        "architecture":"Frozen ImageNet EfficientNet-B0; video mean/std embeddings plus eight texture statistics; fitted regularized logistic head",
        "backbone":"EfficientNet_B0_Weights.IMAGENET1K_V1","feature_schema":"mean1280_std1280_texture8",
        "mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist(),"coef":clf.coef_[0].tolist(),
        "intercept":float(clf.intercept_[0]),"temperature":fitted_temperature,"temperature_fitted":True,
        "calibration_note":"Fitted on a small separate calibration split; not a guarantee of reliable confidence or external accuracy.",
        "display_score":"uncalibrated classifier score; temperature-scaled variant is evaluated separately",
        "training_videos":int(masks["train"].sum()),"validation_videos":int(masks["val"].sum()),
        "calibration_videos":int(masks["calibration"].sum()),"test_videos":int(masks["test"].sum()),
        "train_hashes":usable.loc[masks["train"],"sha256"].tolist(),
        "test_hashes":usable.loc[masks["test"],"sha256"].tolist(),
        "manifest_sha256":sha256(manifest),"config":cfg,"chosen_C":choices[index]["C"],
        "ood_distance_threshold":threshold,"ood_note":"Heuristic distance gate derived only from train/validation features; not validated out-of-distribution detection.",
        "provenance":provenance(),"grouping":"filename source pairs; identity/donor independence unverified",
        "dataset_origin":"User uploads; labels supplied by user; origin/license not independently verified"}
    write_json(output,head)
    p=predict_head(x,head,calibrated=False)
    calibrated_p=predict_head(x,head,calibrated=True)
    test=masks["test"]
    measured=evaluate(y[test],p[test],usable.loc[test,"source_group"].to_numpy(),str(folder/"visual"))
    evaluate(y[test],calibrated_p[test],usable.loc[test,"source_group"].to_numpy(),str(folder/"visual_temperature_scaled"))
    predictions=usable[["file","label","source_group","split","sha256"]].copy()
    predictions["fake_score"]=p;predictions["temperature_scaled_score"]=calibrated_p
    predictions["prediction_at_0_5"]=(p>=.5).astype(int)
    predictions["feature_distance"]=distance
    predictions.to_csv(folder/"all_predictions.csv",index=False)
    predictions[test].to_csv(folder/"heldout_predictions.csv",index=False)
    comparisons=[{"model":"EfficientNet transfer classifier","test":measured}]
    # Baselines use texture only and fit all transforms on training rows only.
    for name,baseline in [("texture_logistic",make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight="balanced",max_iter=2000,random_state=42))),
                          ("texture_forest",RandomForestClassifier(n_estimators=300,max_depth=5,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=4))]:
        baseline.fit(x[masks["train"],-8:],y[masks["train"]])
        probs=baseline.predict_proba(x[test,-8:])[:,1]
        stats=evaluate(y[test],probs,usable.loc[test,"source_group"].to_numpy(),str(folder/name))
        comparisons.append({"model":name,"test":stats})
    excluded=frame.loc[~frame.usable,["file","label","split","quality_reason"]]
    excluded.to_csv(folder/"excluded.csv",index=False)
    write_json(folder/"model_selection.json",{"selection_split":"val only; test not used for tuning","candidates":choices})
    write_json(folder/"comparison.json",comparisons)
    summary={"scope":"video-only visual pilot","received_videos":len(frame),"usable_videos":len(usable),
        "excluded_videos":len(excluded),"test_total":int((frame.split=="test").sum()),
        "test_evaluated":int(test.sum()),"test_coverage":float(test.sum()/max(1,(frame.split=="test").sum())),
        "split_counts":{name:int(mask.sum()) for name,mask in masks.items()},"test_metrics":measured,
        "seconds_including_preprocessing":time.perf_counter()-started,"head_bytes":Path(output).stat().st_size,
        "backbone_bytes":(Path(__import__("torch").hub.get_dir())/"checkpoints/efficientnet_b0_rwightman-7f5810bc.pth").stat().st_size,"model_sha256":sha256(output),
        "limitations":["Small user-labeled dataset","Identity/donor-disjoint splits not verified","External unfamiliar-video accuracy not measured","No perfect accuracy or universal detection claim"]}
    write_json(folder/"summary.json",summary)
    return summary
