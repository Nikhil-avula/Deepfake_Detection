import subprocess
from pathlib import Path
import numpy as np
import pytest
from deepguard.media import probe,sample_video,MediaError,texture_features
from deepguard.utils import config
from deepguard.robustness import perturb
from deepguard.inference import Analyzer


@pytest.fixture
def blank(tmp_path):
    path=tmp_path/'engineering-blank.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i','color=c=blue:s=320x240:r=25:d=2',
                    '-c:v','libx264','-pix_fmt','yuv420p',str(path)],check=True)
    return path


def test_decode_blank_and_abstain_without_fake_score(blank,tmp_path):
    cfg=config();info=probe(blank,cfg)
    assert 1.9<info['duration']<2.2
    f=sample_video(blank,cfg)
    assert len(f['faces'])==0 and not f['usable']
    r,_=Analyzer(cfg,str(tmp_path/'no-model.json')).analyze(blank)
    assert r['decision']=='Setup required' and r['fake_score'] is None


def test_missing_ffprobe_is_setup_error_not_bad_video(blank,tmp_path,monkeypatch):
    monkeypatch.setattr('deepguard.media.shutil.which',lambda name:None)
    r,_=Analyzer(config(),str(tmp_path/'no-model')).analyze(blank)
    assert r['decision']=='Setup required'
    assert 'FFprobe' in r['reasons'][0]
    assert r['fake_score'] is None


def test_missing_model_is_named_as_missing_file(blank,tmp_path):
    r,_=Analyzer(config(),str(tmp_path/'absent.json')).analyze(blank)
    assert r['decision']=='Setup required'
    assert 'checkpoints/visual.json' in r['reasons'][0]


@pytest.mark.parametrize('content',[b'',b'not a video',b'\x00'*256])
def test_empty_corrupt_input_has_actionable_error(tmp_path,content):
    p=tmp_path/'bad.mp4';p.write_bytes(content)
    r,_=Analyzer(config(),str(tmp_path/'no-model')).analyze(p)
    assert r['decision']=='Cannot analyze' and r['reasons']


def test_missing_file_does_not_crash(tmp_path):
    result,_=Analyzer(config(),str(tmp_path/'missing-model')).analyze(tmp_path/'missing-video.mp4')
    assert result['fake_score'] is None


def test_duration_and_file_size_limits(blank):
    cfg=config();cfg['max_duration']=1
    with pytest.raises(MediaError):probe(blank,cfg)
    cfg=config();cfg['max_upload_mb']=.000001
    with pytest.raises(MediaError):probe(blank,cfg)


def test_whole_clip_sampling_shapes_and_texture(blank):
    cfg=config();f=sample_video(blank,cfg)
    assert f['sampled']==cfg['frames']
    assert f['faces'].shape==(0,224,224,3)
    assert texture_features(np.zeros((2,224,224,3),dtype=np.uint8)).shape==(8,)


@pytest.mark.parametrize('kind',['compression','blur','darkness'])
def test_visual_perturbations_decode(blank,tmp_path,kind):
    target=tmp_path/f'{kind}.mp4';perturb(str(blank),str(target),kind)
    assert abs(probe(target,config())['duration']-probe(blank,config())['duration'])<.2
