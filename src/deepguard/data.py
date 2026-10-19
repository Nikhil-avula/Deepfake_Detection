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


def prepare_labeled_dataset(manifest: str, output: str, cfg: dict) -> pd.DataFrame:
    """Verify owner-confirmed labels and bytes before creating grouped splits."""
    frame=pd.read_csv(manifest,keep_default_na=False,dtype={"source_group":str})
    required={"path","file","label","sha256"}
    if not required.issubset(frame.columns):
        raise ValueError("Labeled manifest requires path, file, label and sha256 columns")
    if frame.empty or not frame.label.isin([0,1]).all():
        raise ValueError("Labels must be 0 (Real) or 1 (Fake)")
    frame["label"]=frame.label.astype(int)
    frame["path"]=[str(Path(p).resolve()) for p in frame.path]
    if frame.path.duplicated().any() or frame.file.duplicated().any():
        raise ValueError("Duplicate video paths or filenames in labeled manifest")
    for _,row in frame.iterrows():
        if not re.fullmatch(r"(?:\d{4}(?:_fake)?|v\d+|vs\d+|(?:real_v|fake_vs)\d+_aug_\d+[^/]*)\.mp4",row.file):
            raise ValueError(f"Unexpected uploaded filename: {row.file}")
        if not Path(row.path).is_file() or sha256(row.path)!=row.sha256:
            raise ValueError(f"Missing or changed video: {row.file}")
        if "class_name" in frame and row.class_name!={0:"Real",1:"Fake"}[row.label]:
            raise ValueError(f"Conflicting class name and numeric label: {row.file}")
        info=probe(row.path,cfg)
        for key,value in info.items():frame.loc[row.name,key]=value
    # Recompute conservative grouping instead of trusting a supplied group column.
    frame["source_group"]=[filename_family(name) for name in frame.file]
    conflicts=frame.groupby("sha256").label.nunique().gt(1)
    if conflicts.any():raise ValueError("Identical files have conflicting real/fake labels")
    # Reuploads are receipts, not extra independent training examples.
    frame=frame.sort_values("file").reset_index(drop=True)
    result=split_pairs(frame,cfg["seed"]).drop_duplicates("sha256").reset_index(drop=True)
    assert_no_leakage(result)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output,index=False)
    return result


def filename_family(name: str) -> str:
    """Conservatively group numbered real/fake sources and every augmentation.

    Matching numbers are a possible relationship, not verified donor identities.
    Legacy 0000 clips have a separate namespace from new v1/vs1 clips.
    """
    legacy=re.fullmatch(r"(\d{4})(?:_fake)?\.mp4",name)
    if legacy:return legacy.group(1)
    expanded=re.match(r"(?:(?:real_v|fake_vs)(\d+)_aug_\d+|v(?:s)?(\d+)(?:\.mp4|__))",name)
    if not expanded:raise ValueError(f"Unrecognized source filename: {name}")
    return f"expanded:{int(expanded.group(1) or expanded.group(2)):04}"


def expanded_splits(frame: pd.DataFrame, previous: pd.DataFrame, seed: int=42) -> pd.DataFrame:
    """Preserve old assignments; split only new filename families."""
    result=frame.copy()
    result['source_group']=[filename_family(name) for name in result.file]
    old=previous.assign(source_group=previous.file.map(filename_family)).groupby('source_group').split.first().to_dict()
    fresh=result[~result.source_group.isin(old)].copy()
    assigned=split_pairs(fresh,seed).groupby('source_group').split.first().to_dict() if len(fresh) else {}
    result['split']=result.source_group.map({**old,**assigned})
    cohorts=previous.assign(source_group=previous.file.map(filename_family)).groupby('source_group').evaluation_cohort.first().to_dict() if 'evaluation_cohort' in previous else {g:'historical' for g in old}
    result['evaluation_cohort']=result.source_group.map(cohorts).fillna('new_sources')
    assert_no_leakage(result)
    return result


def augmentation_mask(frame: pd.DataFrame) -> pd.Series:
    """Handle CSV-inferred booleans as well as explicit owner-manifest strings."""
    return frame.get('is_augmented',pd.Series(False,index=frame.index)).astype(str).str.lower().eq('true')


def balanced_eligible(frame: pd.DataFrame, seed: int=42) -> pd.DataFrame:
    """Balance after quality filtering; evaluate originals, never augmentations."""
    result=frame.copy();result['selected']=False;result['selection_reason']='quality excluded'
    augmented=augmentation_mask(result)
    result.loc[result.usable,'selection_reason']='balance reserve'
    for split in ['train','val','calibration','test']:
        for aug in ([False,True] if split=='train' else [False]):
            pool=result[result.usable & result.split.eq(split) & augmented.eq(aug)]
            # Keep historical monitoring separate from the new-source test cohort.
            cohorts=pool.evaluation_cohort.unique() if 'evaluation_cohort' in pool else [None]
            for cohort in cohorts:
                part=pool if cohort is None else pool[pool.evaluation_cohort.eq(cohort)]
                n=min(int(part.label.eq(0).sum()),int(part.label.eq(1).sum()))
                for label in [0,1]:
                    chosen=part[part.label.eq(label)].sort_values('sha256').sample(n=n,random_state=seed).index
                    result.loc[chosen,'selected']=True;result.loc[chosen,'selection_reason']='selected'
    result.loc[result.usable & ~result.split.eq('train') & augmented,'selection_reason']='evaluation augmentation reserved'
    assert_no_leakage(result)
    return result


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
