"""User-labeled visual videos, file integrity and reproducible source-pair splits."""
from __future__ import annotations
import json
import re
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from .media import probe
from .utils import sha256,write_json


def assert_no_leakage(frame: pd.DataFrame) -> None:
    if frame.path.duplicated().any():raise ValueError("Duplicate paths in dataset")
    for key in ["source_group","sha256"]:
        if frame.groupby(key).split.nunique().gt(1).any():
            raise ValueError(f"Data leakage: {key} crosses dataset partitions")


def split_pairs(frame: pd.DataFrame,seed: int=42) -> pd.DataFrame:
    """Keep filename-related pairs and byte-identical copies together.

    This is source-pair grouping, not a certified identity-disjoint benchmark.
    """
    frame=frame.copy().reset_index(drop=True)
    parent={str(g):str(g) for g in frame.source_group.unique()}
    def root(g):
        while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
        return g
    for _,part in frame.groupby("sha256"):
        if part.label.nunique()>1:raise ValueError("Identical files have conflicting real/fake labels")
        groups=list(map(str,part.source_group.unique()))
        for g in groups[1:]:parent[root(g)]=root(groups[0])
    frame["source_group"]=[root(str(g)) for g in frame.source_group]
    groups=np.asarray(sorted(frame.source_group.unique()),dtype=object)
    if len(groups)<12:raise ValueError("At least 12 independent source groups required for this pilot")
    np.random.default_rng(seed).shuffle(groups)
    n=len(groups); train=max(1,int(n*.60));val=max(2,int(n*.13));cal=max(2,int(n*.13))
    partitions={g:("train" if i<train else "val" if i<train+val else "calibration" if i<train+val+cal else "test") for i,g in enumerate(groups)}
    frame["split"]=frame.source_group.map(partitions)
    assert_no_leakage(frame)
    for name in ["train","val","calibration","test"]:
        if set(frame.loc[frame.split==name,"label"])!={0,1}:
            raise ValueError(f"{name} needs both genuine and manipulated videos")
    return frame


def import_uploads(upload_index: str,destination: str,cfg: dict,output: str) -> pd.DataFrame:
    records=json.loads(Path(upload_index).read_text())
    dest=Path(destination).resolve();dest.mkdir(parents=True,exist_ok=True)
    def one(record):
        name=record["name"]
        if not re.fullmatch(r"\d{4}(?:_fake)?\.mp4",name):raise ValueError("Unexpected uploaded filename")
        target=dest/name
        source=Path(record["path"])
        if target.exists() and sha256(target)!=sha256(source):raise ValueError("Existing dataset file differs; refusing overwrite")
        if not target.exists():shutil.copyfile(source,target)
        row={"path":str(target),"file":name,"label":int(record["label"]),
             "source_group":name.split("_")[0].split(".")[0],"sha256":sha256(target),
             "label_source":"user-supplied real/fake naming convention",
             "identity_metadata":"not supplied","dataset":"user-uploaded visual pilot"}
        try:row.update(probe(target,cfg));row["error"]=""
        except ValueError as exc:row["error"]=str(exc)
        return row
    with ThreadPoolExecutor(max_workers=4) as executor:rows=list(executor.map(one,records))
    frame=pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output,index=False)
    write_json(Path(output).with_suffix(".json"),{"files":len(frame),"bytes":int(frame.bytes.sum()) if "bytes" in frame else 0,
        "class_counts":{str(k):int(v) for k,v in frame.label.value_counts().items()},
        "corrupt_files":int(frame.error.ne("").sum()),"unique_hashes":frame.sha256.nunique(),
        "dataset_origin":"Not verified; no named benchmark provenance claimed",
        "grouping":"Matching filename stems are conservatively treated as related originals/derivatives",
        "identity_disjoint":"Not verified; identity/donor metadata was not supplied"})
    return frame
