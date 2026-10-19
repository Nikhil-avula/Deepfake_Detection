import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from deepguard.data import split_pairs,assert_no_leakage
from deepguard.data import filename_family,expanded_splits,balanced_eligible,augmentation_mask


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


def test_augmented_fake_and_real_families_are_grouped():
    assert filename_family('v12.mp4')==filename_family('vs12.mp4')
    assert filename_family('real_v12_aug_4.mp4')==filename_family('fake_vs12_aug_4 - Copy-random.mp4')
    assert filename_family('0012.mp4')!=filename_family('v12.mp4')
    with pytest.raises(ValueError):filename_family('unrelated.mp4')


def test_augmentation_flags_survive_csv_boolean_inference():
    f=pd.DataFrame({'is_augmented':[True,False,None,'true','false','TRUE']})
    assert augmentation_mask(f).tolist()==[True,False,False,True,False,True]


def test_expanded_split_preserves_history_and_is_repeatable():
    old=frame();old['file']=[f'{i:04}'+('_fake' if label else '')+'.mp4' for i in range(49) for label in [0,1]]
    old=split_pairs(old)
    new=[]
    for i in range(1,31):
        for label,prefix in [(0,'v'),(1,'vs')]:
            for aug in [False,True]:
                name=f'{prefix}{i}.mp4' if not aug else f'{"fake_vs" if label else "real_v"}{i}_aug_0.mp4'
                new.append(dict(file=name,path='/'+name,label=label,source_group='untrusted',sha256='hash-'+name))
    result=expanded_splits(pd.concat([old,pd.DataFrame(new)],ignore_index=True),old)
    assert result.iloc[:len(old)].split.tolist()==old.split.tolist()
    assert result.groupby('source_group').split.nunique().eq(1).all()
    again=expanded_splits(result,result)
    pd.testing.assert_frame_equal(result,again)
    assert set(result.evaluation_cohort)=={'historical','new_sources'}


def test_quality_balance_and_original_only_evaluation():
    records=[]
    for split in ['train','val','calibration','test']:
        for label in [0,1]:
            for i in range(4+label):
                for augmented in [False,True]:
                    key=f'{split}-{label}-{i}-{augmented}'
                    records.append(dict(path='/'+key,file=key,sha256=key,source_group=f'{split}-{i}',split=split,label=label,
                        usable=i!=0,is_augmented='true' if augmented else '',evaluation_cohort='new_sources'))
    result=balanced_eligible(pd.DataFrame(records));chosen=result[result.selected]
    assert not result.loc[~result.usable,'selected'].any()
    assert chosen.groupby('split').label.value_counts().unstack().nunique(axis=1).eq(1).all()
    assert not chosen.loc[chosen.split!='train','is_augmented'].eq('true').any()
