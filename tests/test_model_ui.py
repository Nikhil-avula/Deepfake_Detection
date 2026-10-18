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


@pytest.mark.real_data
def test_genuine_heldout_inference_rename_invariance_and_report(tmp_path):
    import pandas as pd
    model=ROOT/'checkpoints/visual.json';manifest=ROOT/'data/manifests/prepared.csv'
    if not model.exists() or not manifest.exists():pytest.skip('Local owner data and trained artifact are not included in Git')
    f=pd.read_csv(manifest);row=f[(f.split=='test') & f.usable].iloc[0]
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
    if not (ROOT/'checkpoints/visual.json').exists():pytest.skip('Real trained artifact unavailable')
    f=pd.read_csv(ROOT/'data/manifests/prepared.csv');row=f[(f.split=='test') & f.usable].iloc[0]
    result,evidence=Analyzer(config(),str(ROOT/'checkpoints/visual.json')).analyze(row.path)
    app=AppTest.from_file('app/main.py',default_timeout=30).run()
    app.session_state['result']=result;app.session_state['evidence']=evidence
    app.sidebar.radio[0].set_value('Your result').run()
    assert not app.exception
    assert len(app.metric)>=3
