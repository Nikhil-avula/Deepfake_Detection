import numpy as np
import pytest
import torch
from deepguard.models import Visual, Audio, Temporal, Synchronizer, reliability_fusion, load_checkpoint
from deepguard.explain import gradcam
from deepguard.utils import seed_all
from deepguard.training import sync_inputs


@pytest.fixture(autouse=True)
def reproducible():
    seed_all(42)


def test_model_shapes_and_gradients():
    visual = Visual().eval()
    face = torch.rand(2,3,64,64)
    assert visual(face).shape == (2,)
    embedding = visual.embed(face).detach()
    temporal = Temporal()
    assert temporal(embedding[None]).shape == (1,)
    audio = Audio()
    before = audio.head.weight.detach().clone()
    optimizer = torch.optim.Adam(audio.parameters(),lr=.01)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(audio(torch.rand(2,1,64,80)), torch.tensor([0.,1.]))
    loss.backward(); optimizer.step()
    assert not torch.equal(before,audio.head.weight)


def test_sync_two_stream_gradient():
    model = Synchronizer()
    mouth, audio = torch.rand(2,3,5,64,64), torch.rand(2,1,5,64,21)
    out = model(mouth,audio)
    assert out.shape == (2,)
    out.sum().backward()
    assert model.mouth[0].weight.grad.abs().sum() > 0
    assert model.audio[0].weight.grad.abs().sum() > 0


def test_sync_offsets_move_audio_not_mouth():
    f = {"mouths":np.stack([np.full((64,64,3),i,dtype=np.uint8) for i in range(20)]),
         "sync_mels":np.stack([np.full((64,21),i,dtype=np.float32) for i in range(20)]),
         "sync_times":np.arange(20)/5}
    m0,a0 = sync_inputs(f)
    m1,a1 = sync_inputs(f,offset=5)
    assert torch.allclose(m0[:len(m1)], m1)
    assert torch.allclose(a1-a0[:len(a1)], torch.full_like(a1,5))


def test_sync_rejects_timestamp_gaps():
    f = {"mouths":np.zeros((5,64,64,3),dtype=np.uint8), "sync_mels":np.zeros((5,64,21)),
         "sync_times":np.arange(5)*2}
    assert sync_inputs(f) is None


def test_quality_fusion_and_missing_evidence():
    p = reliability_fusion(torch.tensor([.9,.1]),torch.tensor([1.,0.]),torch.ones(2))
    assert p.item() == pytest.approx(.9)
    with pytest.raises(ValueError,match="Insufficient"):
        reliability_fusion(torch.ones(2)*.5,torch.zeros(2),torch.zeros(2))


def test_untrained_checkpoint_rejected(tmp_path):
    path = tmp_path/"bad.pt"
    torch.save({"kind":"audio", "state_dict":Audio().state_dict()},path)
    with pytest.raises(ValueError,match="provenance"):
        load_checkpoint(str(path))


def test_gradcam_is_bounded_and_spatial():
    cam = gradcam(Visual(),torch.rand(1,3,64,64))
    assert cam.shape == (1,64,64)
    assert torch.isfinite(cam).all()
    assert cam.min() >= 0 and cam.max() <= 1.00001
