"""Train a small-data visual classifier with independent validation/calibration/test sets."""
from __future__ import annotations
import json
import logging
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .media import sample_video,MediaError
from .models import FaceEncoder,aggregate,cache_key,predict_head,head_logits
from .data import assert_no_leakage,balanced_eligible,augmentation_mask
from .evaluation import score,evaluate,temperature
from .utils import seed_all,sha256,write_json,provenance
from .selection import fit_candidates,group_weights

LOG=logging.getLogger(__name__)


def prepare_features(manifest: str,cfg: dict,cache: str="data/processed") -> pd.DataFrame:
    seed_all(cfg["seed"],cfg["threads"])
    frame=pd.read_csv(manifest,keep_default_na=False,dtype={"source_group":str})
    assert_no_leakage(frame)
    if not frame.label.isin([0,1]).all():raise ValueError("Training labels must be 0 or 1")
    if set(frame.split)!={"train","val","calibration","test"}:
        raise ValueError("Training requires train, val, calibration and test partitions")
    for row in frame.itertuples():
        if not Path(row.path).is_file() or sha256(row.path)!=row.sha256:
            raise ValueError(f"Missing or changed training video: {row.file}")
    encoder=None; rows=[];folder=Path(cache);folder.mkdir(parents=True,exist_ok=True)
    for i,(_,row) in enumerate(frame.iterrows()):
        record=row.to_dict();path=folder/(cache_key(row.path,cfg)+".npz")
        started=time.perf_counter()
        if path.exists():
            with np.load(path,allow_pickle=False) as f:
                usable=bool(f["usable"]);reason=str(f["reason"]);count=len(f["embeddings"])
        else:
            try:media=sample_video(row.path,cfg)
            except MediaError as exc:
                media={'usable':False,'reasons':[str(exc)],'faces':np.empty((0,cfg['image_size'],cfg['image_size'],3),dtype=np.uint8),
                       'times':np.array([]),'sharpness':np.array([]),'sizes':np.array([])}
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
    if cfg.get('balanced_training'):
        frame=balanced_eligible(frame,cfg['seed'])
    else:
        frame['selected']=frame.usable;frame['selection_reason']=np.where(frame.usable,'selected','quality excluded')
    frame.to_csv(Path(manifest).with_name('prepared.csv'),index=False)
    usable=frame[frame.selected].copy().reset_index(drop=True)
    assert_no_leakage(usable)
    for name in ["train","val","calibration","test"]:
        if set(usable.loc[usable.split==name,"label"])!={0,1}:raise ValueError(f"Usable {name} videos must include both classes")
    x=np.stack([np.load(p,allow_pickle=False)["features"] for p in usable.feature_path])
    y=usable.label.to_numpy()
    masks={name:(usable.split==name).to_numpy() for name in ["train","val","calibration","test"]}
    folder=Path(results);folder.mkdir(parents=True,exist_ok=True)
    parameters,choices,histories,index=fit_candidates(x[masks['train']],y[masks['train']],
        usable.loc[masks['train'],'source_group'].to_numpy(),x[masks['val']],y[masks['val']],cfg['seed'])
    calibration_logits=head_logits(x[masks['calibration']],parameters)
    fitted_temperature=temperature(y[masks["calibration"]],calibration_logits)
    z=(x-np.asarray(parameters['mean']))/np.asarray(parameters['scale'])
    reference=masks["train"]|masks["val"]
    distance=np.sqrt(np.mean(z**2,axis=1))
    threshold=float(np.quantile(distance[reference],.99)*1.5)
    head={"schema":2,"kind":"visual","problem_statement":"Classify facial visual evidence in an uploaded video as real or manipulated, with uncertainty and limitations.",
        'trained_at_utc':datetime.now(timezone.utc).isoformat(),
        "architecture":"Frozen ImageNet EfficientNet-B0; video mean/std embeddings plus eight texture statistics; "+choices[index]['model'],
        "backbone":"EfficientNet_B0_Weights.IMAGENET1K_V1","feature_schema":"mean1280_std1280_texture8",
        **parameters,"temperature":fitted_temperature,"temperature_fitted":True,
        "calibration_note":"Fitted on a small separate calibration split; not a guarantee of reliable confidence or external accuracy.",
        "display_score":"uncalibrated classifier score; temperature-scaled variant is evaluated separately",
        "training_videos":int(masks["train"].sum()),"validation_videos":int(masks["val"].sum()),
        "calibration_videos":int(masks["calibration"].sum()),"test_videos":int(masks["test"].sum()),
        "train_hashes":usable.loc[masks["train"],"sha256"].tolist(),
        'validation_hashes':usable.loc[masks['val'],'sha256'].tolist(),
        'calibration_hashes':usable.loc[masks['calibration'],'sha256'].tolist(),
        "test_hashes":usable.loc[masks["test"],"sha256"].tolist(),
        "manifest_sha256":sha256(manifest),"config":cfg,"chosen_C":choices[index].get('C'),"selected_classifier":choices[index],
        "ood_distance_threshold":threshold,"ood_note":"Heuristic distance gate derived only from train/validation features; not validated out-of-distribution detection.",
        "provenance":provenance(),"grouping":"filename source pairs; identity/donor independence unverified",
        "dataset_origin":"User uploads; labels supplied by user; origin/license not independently verified"}
    write_json(output,head)
    p=predict_head(x,head,calibrated=False)
    calibrated_p=predict_head(x,head,calibrated=True)
    all_test=masks['test'].copy()
    test=all_test & usable.evaluation_cohort.eq('new_sources').to_numpy() if cfg.get('balanced_training') else all_test
    measured=evaluate(y[test],p[test],usable.loc[test,"source_group"].to_numpy(),str(folder/"visual"))
    evaluate(y[test],calibrated_p[test],usable.loc[test,"source_group"].to_numpy(),str(folder/"visual_temperature_scaled"))
    historical=all_test & ~test
    if historical.any():evaluate(y[historical],p[historical],usable.loc[historical,'source_group'].to_numpy(),str(folder/'historical_monitoring'))
    predictions=usable[["file","label","source_group","split","sha256"]].copy()
    predictions["fake_score"]=p;predictions["temperature_scaled_score"]=calibrated_p
    predictions["prediction_at_0_5"]=(p>=.5).astype(int)
    predictions["feature_distance"]=distance
    if 'evaluation_cohort' in usable:predictions['evaluation_cohort']=usable.evaluation_cohort
    predictions.to_csv(folder/"all_predictions.csv",index=False)
    predictions[test].to_csv(folder/"heldout_predictions.csv",index=False)
    comparisons=[{"model":"EfficientNet transfer classifier","test":measured}]
    # Baselines use texture only and fit all transforms on training rows only.
    for name,baseline in [("texture_logistic",make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight="balanced",max_iter=2000,random_state=42))),
                          ("texture_forest",RandomForestClassifier(n_estimators=300,max_depth=5,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=4))]:
        weights=group_weights(y[masks['train']],usable.loc[masks['train'],'source_group'].to_numpy())
        if name=='texture_logistic':baseline.fit(x[masks['train'],-8:],y[masks['train']],logisticregression__sample_weight=weights)
        else:baseline.fit(x[masks['train'],-8:],y[masks['train']],sample_weight=weights)
        probs=baseline.predict_proba(x[test,-8:])[:,1]
        stats=evaluate(y[test],probs,usable.loc[test,"source_group"].to_numpy(),str(folder/name))
        comparisons.append({"model":name,"test":stats})
    excluded=frame.loc[~frame.usable,["file","label","split","quality_reason"]]
    excluded.to_csv(folder/"excluded.csv",index=False)
    write_json(folder/"model_selection.json",{"selection_split":"val only; test not used for tuning","candidates":choices,'chosen_index':index,'neural_training_history':histories})
    membership=['file','label','source_group','split','usable','selected','selection_reason']
    membership += [k for k in ['evaluation_cohort','is_augmented'] if k in frame]
    frame[membership].to_csv(folder/'split_membership.csv',index=False)
    frame.loc[~frame.selected,membership].to_csv(folder/'selection_exclusions.csv',index=False)
    write_json(folder/"comparison.json",comparisons)
    original=~augmentation_mask(frame)
    test_pool=frame.split.eq('test') & original
    if cfg.get('balanced_training'):test_pool &= frame.evaluation_cohort.eq('new_sources')
    summary={"scope":"video-only visual pilot","received_videos":len(frame),"usable_videos":int(frame.usable.sum()),
        'selected_videos':len(usable),'balance_reserved_videos':int((frame.usable & ~frame.selected).sum()),
        'classifier':choices[index]['model'],'evaluation_scope':'new-source original videos only' if cfg.get('balanced_training') else 'source-pair test videos',
        'balanced_class_counts':{name:{str(k):int(v) for k,v in usable[usable.split==name].label.value_counts().items()} for name in masks},
        'primary_test_class_counts':{str(k):int(v) for k,v in usable.loc[test,'label'].value_counts().items()},
        'training_original_class_counts':{str(k):int(v) for k,v in usable.loc[masks['train'] & ~augmentation_mask(usable).to_numpy(),'label'].value_counts().items()},
        'training_augmented_class_counts':{str(k):int(v) for k,v in usable.loc[masks['train'] & augmentation_mask(usable).to_numpy(),'label'].value_counts().items()},
        "excluded_videos":len(excluded),"test_total":int(test_pool.sum()),
        "test_evaluated":int(test.sum()),"test_coverage":float(test.sum()/max(1,test_pool.sum())),
        "split_counts":{name:int(mask.sum()) for name,mask in masks.items()},"test_metrics":measured,
        "seconds_including_preprocessing":time.perf_counter()-started,"head_bytes":Path(output).stat().st_size,
        "backbone_bytes":(Path(__import__("torch").hub.get_dir())/"checkpoints/efficientnet_b0_rwightman-7f5810bc.pth").stat().st_size,"model_sha256":sha256(output),
        "limitations":["Small user-labeled dataset","Identity/donor-disjoint splits not verified","External unfamiliar-video accuracy not measured","No perfect accuracy or universal detection claim"]}
    write_json(folder/"summary.json",summary)
    return summary
