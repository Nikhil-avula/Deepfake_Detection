import numpy as np
import pandas as pd
import pytest
from deepguard.data import split_manifest, assert_no_leakage, read_manifest
from deepguard.evaluation import score, fit_temperature, evaluate


def test_connected_derivatives_and_all_identities_stay_together():
    rows = [{"path": f"p{i}", "identities": f"id{i}", "sources": f"s{i}", "label": i%2} for i in range(20)]
    rows += [{"path":"derivative", "identities":"id0|id1", "sources":"s0|s1", "label":1}]
    split = split_manifest(pd.DataFrame(rows))
    assert_no_leakage(split)
    assert len(set(split.loc[[0,1,20], "split"])) == 1
    assert set(split.split) == {"train", "val", "fusion", "calibration", "test"}
    pd.testing.assert_frame_equal(split, split_manifest(pd.DataFrame(rows)))


def test_reject_leakage():
    f = pd.DataFrame([{"path":"a", "identities":"a|b", "sources":"s1", "split":"train"},
                      {"path":"b", "identities":"b", "sources":"s2", "split":"test"}])
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(f)


def test_reject_too_few_components():
    with pytest.raises(ValueError, match="five"):
        split_manifest(pd.DataFrame([{"identities":"x", "sources":"s"}]))


def test_metric_values_and_plots(tmp_path):
    y, p = np.array([0,0,1,1]), np.array([.1,.8,.4,.9])
    result = evaluate(y, p, str(tmp_path))
    assert result["accuracy"] == .5
    assert result["confusion_matrix"] == [[1,1],[1,1]]
    assert result["false_positive_rate"] == .5
    assert result["false_negative_rate"] == .5
    assert result["roc_auc"] == .75
    assert (tmp_path/"curves.png").is_file()


def test_single_class_auc_is_undefined():
    assert score(np.array([0,0]), np.array([.2,.3]))["roc_auc"] is None


@pytest.mark.parametrize("p", [[float("nan")], [1.1], [-.2]])
def test_invalid_probabilities(p):
    with pytest.raises(ValueError):
        score(np.array([1]), np.array(p))


def test_calibration_reduces_overconfident_loss():
    y = np.array([0,1,0,1,0,1,0,1])
    logits = np.array([-10,10,10,-10,-10,10,-10,10])
    temperature = fit_temperature(y, logits)
    def loss(t): return np.mean(np.logaddexp(0, logits/t)-y*logits/t)
    assert loss(temperature) < loss(1)


def test_labels_not_inferred_from_overall(tmp_path):
    path = tmp_path/"file.mp4"; path.write_bytes(b"x")
    csv = tmp_path/"manifest.csv"
    pd.DataFrame([dict(path=str(path), label=1, visual_label=0, audio_label=1,
                       identities="id1", sources="s1", dataset="authorized", method="audio")]).to_csv(csv,index=False)
    f = read_manifest(csv)
    assert f.iloc[0].visual_label == 0 and f.iloc[0].audio_label == 1
