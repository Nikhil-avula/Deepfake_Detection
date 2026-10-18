import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from deepguard.data import split_pairs,assert_no_leakage


def frame():
    return pd.DataFrame([dict(path=f"/{i}-{label}",source_group=f"{i:04}",sha256=f"hash-{i}-{label}",label=label)
                         for i in range(49) for label in [0,1]])


def test_pairs_split_counts_and_repeatability():
    original=frame();split=split_pairs(original)
    assert_no_leakage(split)
    assert split.groupby('source_group').split.nunique().eq(1).all()
    assert split.split.value_counts().to_dict()=={'train':58,'test':16,'val':12,'calibration':12}
    pd.testing.assert_frame_equal(split,split_pairs(original))


def test_same_bytes_merge_groups():
    f=frame();f.loc[2,'sha256']=f.loc[0,'sha256']
    result=split_pairs(f)
    assert result.loc[0,'source_group']==result.loc[2,'source_group']
    assert_no_leakage(result)


def test_conflicting_identical_labels_rejected():
    f=frame();f.loc[1,'sha256']=f.loc[0,'sha256']
    with pytest.raises(ValueError,match='conflicting'):split_pairs(f)


def test_leakage_rejected():
    f=split_pairs(frame());f.loc[1,'split']='test' if f.loc[0,'split']!='test' else 'train'
    with pytest.raises(ValueError,match='leakage'):assert_no_leakage(f)


def test_too_few_groups_rejected():
    with pytest.raises(ValueError,match='12'):split_pairs(frame().iloc[:4])


@pytest.mark.parametrize('fault',[None,'changed_bytes','bad_label','class_conflict'])
def test_confirmed_manifest_integrity_before_training(tmp_path,monkeypatch,fault):
    from deepguard.data import prepare_labeled_dataset
    from deepguard.utils import sha256
    monkeypatch.setattr('deepguard.data.probe',lambda path,cfg:{'duration':1.0})
    rows=[]
    for i in range(49):
        for label in [0,1]:
            name=f'{i:04}'+('_fake' if label else '')+'.mp4'
            path=tmp_path/name;path.write_bytes(f'{i}-{label}'.encode())
            rows.append(dict(path=str(path),file=name,label=label,sha256=sha256(path),
                             class_name='Fake' if label else 'Real',source_group='untrusted'))
    if fault=='changed_bytes':Path(rows[0]['path']).write_bytes(b'changed')
    if fault=='bad_label':rows[0]['label']=2
    if fault=='class_conflict':rows[0]['class_name']='Fake'
    manifest=tmp_path/'labels.csv';pd.DataFrame(rows).to_csv(manifest,index=False)
    output=tmp_path/'splits.csv'
    if fault:
        with pytest.raises(ValueError):prepare_labeled_dataset(str(manifest),str(output),{'seed':42})
        assert not output.exists()
    else:
        result=prepare_labeled_dataset(str(manifest),str(output),{'seed':42})
        assert_no_leakage(result)
        assert result.source_group.nunique()==49
        assert len(result)==98 and output.exists()
