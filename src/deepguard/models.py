"""Trainable visual, audio, temporal and audio-visual synchronization networks."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class Visual(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        self.encoder.classifier = nn.Identity()
        self.head = nn.Linear(1280, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        # Inputs are RGB in [0,1]; use pretrained ImageNet normalization consistently.
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        return self.encoder((x - mean) / std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x)).squeeze(-1)


class Audio(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                     nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                     nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)


class Temporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(1280, 128, batch_first=True)
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.rnn(x)
        return self.head(hidden[-1]).squeeze(-1)


class Synchronizer(nn.Module):
    """Learn an aligned-pair vs time-shifted-pair task; NOT a deepfake classifier.

    Five frames and five audio windows encode local dynamics in a two-stream model.
    Training uses real face/real audio only; offsets are generated within a recording.
    """
    def __init__(self):
        super().__init__()
        self.mouth = nn.Sequential(nn.Conv3d(3, 16, (3, 3, 3), padding=1), nn.ReLU(),
                                   nn.MaxPool3d((1, 2, 2)), nn.Conv3d(16, 32, 3, padding=1),
                                   nn.ReLU(), nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(32, 64))
        self.audio = nn.Sequential(nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(),
                                   nn.AdaptiveAvgPool3d((1, 8, 8)), nn.Flatten(), nn.Linear(1024, 64))
        self.log_scale = nn.Parameter(torch.tensor(1.))
        self.bias = nn.Parameter(torch.tensor(0.))

    def forward(self, mouth: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        similarity = F.cosine_similarity(self.mouth(mouth), self.audio(audio))
        return similarity * self.log_scale.exp().clamp(max=100) + self.bias


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Four probabilities, four quality measures, four availability masks.
        self.net = nn.Sequential(nn.Linear(12, 16), nn.ReLU(), nn.Dropout(.1), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_model(kind: str, pretrained: bool = False) -> nn.Module:
    constructors = {"visual": lambda: Visual(pretrained), "audio": Audio,
                    "temporal": Temporal, "sync": Synchronizer, "fusion": Fusion}
    if kind not in constructors:
        raise ValueError(f"Unknown model kind: {kind}")
    return constructors[kind]()


def load_checkpoint(path: str, device: str = "cpu") -> tuple[nn.Module, dict]:
    # No pickle fallback: checkpoint format restricted to tensors and primitives.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("schema") != 1 or checkpoint.get("trained_samples", 0) < 1:
        raise ValueError("Missing training provenance; refusing checkpoint")
    model = make_model(checkpoint["kind"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def reliability_fusion(probabilities: torch.Tensor, quality: torch.Tensor,
                       available: torch.Tensor) -> torch.Tensor:
    """Quality-weighted log-odds; hypotheses about quality must be evaluated."""
    weights = quality.clamp(0, 1) * available
    if torch.any(weights.sum(-1) <= 0):
        raise ValueError("Insufficient Evidence")
    logits = torch.logit(probabilities.clamp(.0001, .9999))
    return torch.sigmoid((weights * logits).sum(-1) / weights.sum(-1))


def disagreement(probabilities: dict[str, float]) -> float | None:
    if "visual" not in probabilities or "audio" not in probabilities:
        return None
    return abs(probabilities["visual"] - probabilities["audio"])
