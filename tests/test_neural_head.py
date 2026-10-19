import json
import numpy as np
import pytest
import torch
from deepguard.models import predict_head,load_head
from deepguard.selection import group_weights,fit_candidates
from deepguard.evaluation import score


def test_neural_export_matches_torch_and_rejects_malformed_parameters(tmp_path):
    torch.manual_seed(7)
    layer=torch.nn.Linear(2568,4);output=torch.nn.Linear(4,1)
    features=np.random.default_rng(7).normal(size=(3,2568)).astype('float32')
    head=dict(schema=2,kind='visual',training_videos=12,classifier_type='mlp',mean=[0.]*2568,scale=[1.]*2568,
              hidden_weight=layer.weight.detach().numpy().T.tolist(),hidden_bias=layer.bias.detach().numpy().tolist(),
              coef=output.weight.detach().numpy().ravel().tolist(),intercept=float(output.bias.detach()[0]),temperature=1.)
    with torch.inference_mode():expected=torch.sigmoid(output(torch.relu(layer(torch.from_numpy(features))))).numpy().ravel()
    path=tmp_path/'head.json';path.write_text(json.dumps(head))
    assert predict_head(features,load_head(path),calibrated=False)==pytest.approx(expected,abs=1e-6)
    head['hidden_bias']=[0.];path.write_text(json.dumps(head))
    with pytest.raises(ValueError,match='neural'):load_head(path)


def test_source_family_weights_do_not_multiply_augmented_evidence():
    labels=np.array([0,0,0,1,1,1]);groups=['a','a','b','c','c','d']
    weights=group_weights(labels,groups)
    assert weights[0]+weights[1]==pytest.approx(weights[2])
    assert weights[3]+weights[4]==pytest.approx(weights[5])
    assert weights[labels==0].sum()==pytest.approx(weights[labels==1].sum())


def test_candidate_selection_uses_validation_scores_and_exports_fitted_parameters():
    torch.set_num_threads(2)
    rng=np.random.default_rng(3)
    x=rng.normal(size=(24,2568));y=np.tile([0,1],12)
    groups=np.arange(16).astype(str)
    head,candidates,histories,index=fit_candidates(x[:16],y[:16],groups,x[16:],y[16:],42)
    expected=max(range(len(candidates)),key=lambda i:(candidates[i]['validation']['roc_auc'],-candidates[i]['validation']['brier']))
    assert index==expected and len(candidates)==7
    assert len(histories)==2 and all(h['epochs'] for h in histories)
    observed=score(y[16:],predict_head(x[16:],head,calibrated=False))
    assert observed['roc_auc']==pytest.approx(candidates[index]['validation']['roc_auc'])
