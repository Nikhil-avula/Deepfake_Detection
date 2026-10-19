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


def test_damaged_model_has_setup_guidance(tmp_path):
    model=tmp_path/'model.json';model.write_text('{bad json')
    result,_=Analyzer(config(),str(model)).analyze(tmp_path/'any.mp4')
    assert result['decision']=='Setup required' and result['fake_score'] is None
    assert 'damaged' in result['reasons'][0]


@pytest.mark.parametrize("failure",[OSError("offline"),__import__("http.client",fromlist=["IncompleteRead"]).IncompleteRead(b"partial",100)])
def test_encoder_download_failure_is_setup_error(tmp_path,monkeypatch,failure):
    import json
    cfg=config();model=tmp_path/'model.json'
    model.write_text(json.dumps(dict(schema=2,kind='visual',training_videos=2,mean=[0.]*2568,scale=[1.]*2568,
        coef=[0.]*2568,intercept=0.,config=cfg,ood_distance_threshold=2.)))
    media=dict(usable=True,metadata={},sampled=16,faces=np.zeros((16,224,224,3),dtype=np.uint8),missing=0,multiple=0)
    monkeypatch.setattr('deepguard.inference.sample_video',lambda path,cfg:media)
    def offline():raise failure
    monkeypatch.setattr('deepguard.inference.FaceEncoder',offline)
    result,_=Analyzer(cfg,str(model)).analyze(tmp_path/'sample.mp4')
    assert result['decision']=='Setup required' and result['fake_score'] is None
    assert 'internet' in result['reasons'][0]


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


def test_detector_setup_failure_never_opens_video(monkeypatch,tmp_path):
    from unittest.mock import Mock
    from deepguard.media import SetupError
    cfg=config();cfg['face_detector']='haar'
    monkeypatch.setattr('deepguard.media.probe',lambda *args:dict(duration=1,fps=25))
    monkeypatch.delattr('deepguard.media.cv2.CascadeClassifier')
    capture=Mock()
    monkeypatch.setattr('deepguard.media.cv2.VideoCapture',capture)
    with pytest.raises(SetupError,match='OpenCV'):
        sample_video(tmp_path/'video.mp4',cfg)
    capture.assert_not_called()


def test_video_handle_released_when_face_detection_fails(monkeypatch,tmp_path):
    from unittest.mock import Mock
    cfg=config();cfg['face_detector']='haar'
    monkeypatch.setattr('deepguard.media.probe',lambda *args:dict(duration=1,fps=25))
    capture=Mock();capture.read.return_value=(True,np.zeros((64,64,3),dtype=np.uint8))
    monkeypatch.setattr('deepguard.media.cv2.VideoCapture',lambda *args:capture)
    def fail(*args):raise RuntimeError('detector failed during decoding')
    monkeypatch.setattr('deepguard.media.detect_boxes',fail)
    with pytest.raises(RuntimeError,match='during decoding'):
        sample_video(tmp_path/'video.mp4',cfg)
    capture.release.assert_called_once()


def test_foreground_face_selection_keeps_dominant_person_only():
    from deepguard.media import primary_face
    foreground=[10,10,100,100];background=[150,10,30,30]
    assert primary_face([background,foreground],4)==foreground
    assert primary_face([[0,0,80,80],foreground],4) is None
    assert primary_face([],4) is None
    assert primary_face([foreground],4)==foreground


def test_dominant_face_policy_changes_feature_cache_key(tmp_path):
    from deepguard.models import cache_key
    video=tmp_path/'sample.mp4';video.write_bytes(b'cache-test')
    old=config();new={**old,'dominant_face_area_ratio':4.}
    assert cache_key(video,old)!=cache_key(video,new)


def test_sequential_sampler_reads_full_clip_without_seeking():
    from unittest.mock import Mock
    from deepguard.media import sampled_frames
    capture=Mock();capture.grab.return_value=True
    capture.retrieve.return_value=(True,np.zeros((2,2,3),dtype=np.uint8))
    result=list(sampled_frames(capture,[0.,.5,1.],10,'sequential_v1'))
    assert [t for t,_ in result]==[0.,.5,1.]
    assert capture.grab.call_count==11
    assert capture.retrieve.call_count==3
    capture.set.assert_not_called()
    capture.read.assert_not_called()


def test_sequential_sampler_decodes_real_clip(blank):
    cfg={**config(),'video_sampler':'sequential_v1','frames':24}
    media=sample_video(blank,cfg)
    assert media['sampled']==24 and not media['usable']
    assert len(media['faces'])==0


def test_trained_model_never_classifies_a_face_free_video(blank):
    from deepguard.utils import ROOT
    model=ROOT/'checkpoints/visual.json'
    if not model.exists():pytest.skip('Trained artifact unavailable')
    report,media=Analyzer(config(),str(model)).analyze(blank)
    assert report['decision']=='Cannot analyze'
    assert report['fake_score'] is None
    assert media['sampled']>0 and report['faces_found']==0
    assert any('Found 0 usable face samples' in reason for reason in report['reasons'])
