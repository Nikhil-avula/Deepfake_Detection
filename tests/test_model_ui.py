import json
import shutil
from pathlib import Path
import numpy as np
import pytest
import torch
from deepguard.models import FaceEncoder,aggregate,load_head,predict_head
from deepguard.utils import ROOT,config,seed_all
from deepguard.inference import Analyzer
from deepguard.reporting import html_report
from deepguard.utils import sha256


def test_encoder_and_aggregate_shapes():
    seed_all();encoder=FaceEncoder(pretrained=False)
    face=np.zeros((2,224,224,3),dtype=np.uint8)
    embedding=encoder.encode(face)
    assert embedding.shape==(2,1280)
    assert aggregate(embedding,face).shape==(2568,)


def test_untrained_or_malformed_checkpoint_rejected(tmp_path):
    p=tmp_path/'head.json';p.write_text(json.dumps({'schema':2,'kind':'visual','training_videos':0}))
    with pytest.raises(ValueError):load_head(p)


def test_report_escapes_untrusted_text():
    report=html_report({'decision':'Cannot analyze','reason':'<script>alert(1)</script>'})
    assert '<script>' not in report and '&lt;script&gt;' in report


def test_all_six_ui_pages_render_without_exception():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file('app/main.py',default_timeout=30).run()
    assert not app.exception
    for page in ['Check a video','Your result','How reliable is it?','Our dataset','About this project','Start']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception,page


def test_clear_button_removes_retained_result():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file('app/main.py',default_timeout=30).run()
    app.session_state['result']={'decision':'Cannot analyze','fake_score':None,'reasons':[],'seconds':0,'membership':'unknown'}
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert 'result' not in app.session_state.filtered_state


def test_published_model_and_dataset_reports_are_consistent():
    summary=json.loads((ROOT/'reports/results/summary.json').read_text())
    inventory=json.loads((ROOT/'experiments/dataset_inventory.json').read_text())
    head=load_head(ROOT/'checkpoints/visual.json')
    assert sum(inventory['class_counts'].values())==summary['received_videos']
    assert sha256(ROOT/'checkpoints/visual.json')==summary['model_sha256']
    assert head['training_videos']==summary['split_counts']['train']
    if head['config'].get('balanced_training'):
        for counts in summary['balanced_class_counts'].values():assert counts['0']==counts['1']
        assert sum(summary['split_counts'].values())==summary['selected_videos']


@pytest.mark.real_data
@pytest.mark.parametrize('label',[0,1])
def test_genuine_heldout_inference_rename_invariance_and_report(tmp_path,label):
    import pandas as pd
    model=ROOT/'checkpoints/visual.json';manifest=ROOT/'data/manifests/prepared.csv'
    if not model.exists() or not manifest.exists():pytest.skip('Local owner data and trained artifact are not included in Git')
    f=pd.read_csv(manifest)
    selected=f.get('selected',f.usable)
    cohort=f.evaluation_cohort.eq('new_sources') if 'evaluation_cohort' in f else True
    row=f[(f.split=='test') & f.usable & selected & cohort & f.label.eq(label)].iloc[0]
    analyzer=Analyzer(config(),str(model))
    first,evidence=analyzer.analyze(row.path)
    assert first['fake_score'] is not None and len(evidence['faces'])>=4
    renamed=tmp_path/'arbitrary-name_fake.mp4';shutil.copyfile(row.path,renamed)
    second,_=analyzer.analyze(renamed)
    assert first['fake_score']==pytest.approx(second['fake_score'],abs=1e-6)
    assert first['membership']=='held-out test video'
    assert first['decision'] in {'Likely real','Likely fake','Not sure'}
    assert '<html' in html_report(first)


@pytest.mark.real_data
def test_ui_renders_actual_inference_result():
    from streamlit.testing.v1 import AppTest
    import pandas as pd
    if not (ROOT/'checkpoints/visual.json').exists() or not (ROOT/'data/manifests/prepared.csv').exists():
        pytest.skip('Local owner media and prepared manifest unavailable')
    f=pd.read_csv(ROOT/'data/manifests/prepared.csv')
    selected=f.get('selected',f.usable)
    cohort=f.evaluation_cohort.eq('new_sources') if 'evaluation_cohort' in f else True
    row=f[(f.split=='test') & f.usable & selected & cohort].iloc[0]
    result,evidence=Analyzer(config(),str(ROOT/'checkpoints/visual.json')).analyze(row.path)
    app=AppTest.from_file('app/main.py',default_timeout=30).run()
    app.session_state['result']=result;app.session_state['evidence']=evidence
    app.sidebar.radio[0].set_value('Your result').run()
    assert not app.exception
    assert len(app.metric)>=3


def test_upload_failure_exposes_diagnostic_and_marks_status_failed():
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest
    from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
    upload=UploadedFile(UploadedFileRec('error-case','video.mp4','video/mp4',b'test'),None)
    app=AppTest.from_file('app/main.py',default_timeout=30).run()
    app.sidebar.radio[0].set_value('Check a video')
    with patch('streamlit.file_uploader',return_value=upload), patch.object(Analyzer,'analyze',side_effect=RuntimeError('diagnostic regression')):
        app.run()
        next(b for b in app.button if b.label=='Check this video').click().run()
    assert not app.exception
    assert any('could not finish' in e.value for e in app.error)
    assert any('RuntimeError: diagnostic regression' in block.value for block in app.code)
    assert app.status[0].state=='error'
    assert 'result' not in app.session_state.filtered_state


def test_retry_only_runs_for_insufficient_evidence(monkeypatch,tmp_path):
    analyzer=Analyzer(config(),str(ROOT/'checkpoints/visual.json'))
    analyzer.cfg={**analyzer.cfg,'retry_model':'unused.json'}
    class Retry:
        calls=0
        def analyze(self,path):
            self.calls+=1
            return dict(decision='Likely fake',fake_score=.8,reasons=[],seconds=.1),{'faces':[]}
    retry=Retry();analyzer.retry=retry
    monkeypatch.setattr(analyzer,'_analyze',lambda path:({'decision':'Likely real','fake_score':.2,'reasons':[],'seconds':0},None))
    result,_=analyzer.analyze(tmp_path/'video.mp4')
    assert result['fake_score']==.2 and retry.calls==0
    monkeypatch.setattr(analyzer,'_analyze',lambda path:({'decision':'Cannot analyze','fake_score':None,'reasons':['too few faces'],'faces_found':2,'seconds':0},None))
    result,_=analyzer.analyze(tmp_path/'video.mp4')
    assert retry.calls==1 and result['fake_score']==.8
    assert result['analysis_pass']=='additional face search' and result['first_pass_faces']==2


def test_missing_retry_model_is_setup_issue(monkeypatch,tmp_path):
    analyzer=Analyzer(config(),str(ROOT/'checkpoints/visual.json'))
    analyzer.cfg={**analyzer.cfg,'retry_model':str(tmp_path/'missing.json')}
    monkeypatch.setattr(analyzer,'_analyze',lambda path:({'decision':'Cannot analyze','fake_score':None,'reasons':[],'seconds':0},None))
    result,_=analyzer.analyze(tmp_path/'video.mp4')
    assert result['decision']=='Setup required' and result['fake_score'] is None


@pytest.mark.real_data
@pytest.mark.parametrize('filename',['v44.mp4','vs44.mp4'])
def test_additional_search_recovers_previously_rejected_face_videos(filename):
    import pandas as pd
    manifest=ROOT/'data/manifests/prepared.csv'
    if not manifest.exists():pytest.skip('Private owner videos unavailable')
    frame=pd.read_csv(manifest)
    row=frame[frame.file.eq(filename)].iloc[0]
    if not Path(row.path).exists():pytest.skip('Private owner video unavailable')
    result,media=Analyzer(config(),str(ROOT/'checkpoints/visual.json')).analyze(row.path)
    assert result['analysis_pass']=='additional face search'
    assert result['fake_score'] is not None and np.isfinite(result['fake_score'])
    assert len(media['faces'])>=4
    assert result['decision'] in {'Likely real','Likely fake','Not sure'}
