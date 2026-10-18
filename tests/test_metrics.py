import numpy as np
import pytest
from deepguard.evaluation import score,temperature,evaluate


def test_known_answer_metrics():
    r=score([0,0,1,1],[.1,.8,.4,.9])
    assert r['accuracy']==.5 and r['confusion_matrix']==[[1,1],[1,1]]
    assert r['roc_auc']==.75 and r['false_positive_rate']==.5 and r['false_negative_rate']==.5
    assert r['brier']==pytest.approx((.01+.64+.36+.01)/4)


def test_single_class_auc_is_undefined():
    assert score([0,0],[.1,.2])['roc_auc'] is None


@pytest.mark.parametrize('p',[[float('nan')],[float('inf')],[1.1],[-.1]])
def test_invalid_scores_rejected(p):
    with pytest.raises(ValueError):score([1],p)


@pytest.mark.parametrize('y,p',[([],[]),([0],[.1,.2]),([2],[.4])])
def test_invalid_targets_rejected(y,p):
    with pytest.raises(ValueError):score(y,p)


def test_calibration_uses_both_classes_and_lowers_its_own_loss():
    y=np.array([0,1,0,1,0,1]);z=np.array([-10,10,10,-10,-10,10])
    t=temperature(y,z)
    loss=lambda a:np.mean(np.logaddexp(0,z/a)-y*z/a)
    assert loss(t)<loss(1)
    with pytest.raises(ValueError):temperature([0,0],[-1,-2])


def test_group_bootstrap_and_plots_are_real_outputs(tmp_path):
    r=evaluate(np.array([0,1,0,1,0,1]),np.array([.1,.9,.2,.8,.3,.7]),np.array(['a','a','b','b','c','c']),str(tmp_path))
    assert r['source_groups']==3
    assert (tmp_path/'evaluation.png').stat().st_size>1000
    assert r['accuracy_group_bootstrap_95_ci']==[1.,1.]
