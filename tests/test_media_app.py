import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
from deepguard.media import probe, preprocess, audio_window, handcrafted
from deepguard.inference import Analyzer
from deepguard.reporting import html_report
from deepguard.utils import config


@pytest.fixture
def media(tmp_path):
    output = tmp_path/"engineering-fixture.mp4"
    subprocess.run(["ffmpeg","-nostdin","-v","error","-y","-f","lavfi","-i","color=c=blue:s=160x120:r=25:d=5",
                    "-f","lavfi","-i","sine=frequency=440:sample_rate=16000:duration=5",
                    "-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-shortest",str(output)],check=True)
    return output


def test_real_decode_no_face_no_prediction_cache_and_report(media,tmp_path):
    cfg = config()
    f = preprocess(media,cfg,tmp_path/"cache")
    assert probe(media)["has_audio"]
    assert len(f["faces"]) == 0
    assert f["audio_mels"].shape[1] == cfg["n_mels"]
    assert len(handcrafted(f)) == 14
    again = preprocess(media,cfg,tmp_path/"cache")
    np.testing.assert_array_equal(f["audio_mels"],again["audio_mels"])
    report,_ = Analyzer(cfg,str(tmp_path/"no-weights")).analyze(str(media))
    assert report["decision"] == "Insufficient Evidence"
    assert report["probabilities"] == {}
    assert "Model not trained" in report["reasons"]
    assert "Insufficient Evidence" in html_report(report)


def test_no_audio_is_explicit(media,tmp_path):
    silent = tmp_path/"silent.mp4"
    subprocess.run(["ffmpeg","-nostdin","-v","error","-y","-i",str(media),"-an","-c:v","copy",str(silent)],check=True)
    f = preprocess(silent,config())
    assert f["audio_mels"].size == 0
    assert float(f["audio_quality"]) == 0


def test_corrupt_and_size_limits(tmp_path,media):
    bad = tmp_path/"bad.mp4"; bad.write_text("not video")
    with pytest.raises(ValueError): probe(bad)
    cfg = config(); cfg["max_duration"] = 1
    with pytest.raises(ValueError,match="limit"):
        preprocess(media,cfg)


def test_audio_timestamp_alignment():
    audio = np.arange(32000)
    np.testing.assert_equal(audio_window(audio,1,.2,16000),audio[14400:17600])
    assert audio_window(audio,0,.2,16000) is None


def test_html_escapes_untrusted_text():
    report = html_report({"input":"<script>alert(1)</script>"})
    assert "<script>" not in report and "&lt;script&gt;" in report


@pytest.mark.parametrize("kind",["compression","blur","lighting","noise"])
def test_perturbation_decodes_and_preserves_duration(media,tmp_path,kind):
    from deepguard.robustness import perturb
    output = tmp_path/f"{kind}.mp4"
    perturb(str(media),str(output),kind)
    info = probe(output)
    assert info["has_audio"]
    assert abs(info["duration"]-probe(media)["duration"]) < .2


def test_streamlit_all_navigation_pages():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file("app/main.py",default_timeout=30).run()
    assert not app.exception
    for page in ["Analyze Video","Detection Results","Forensic Evidence","Research Laboratory","Dataset Explorer","Model Information","About and Documentation"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
