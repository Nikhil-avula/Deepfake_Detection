"""Grad-CAM for the visual classifier; an attribution is not proof of forgery."""
from __future__ import annotations
import torch
from torch.nn import functional as F


def gradcam(model: torch.nn.Module, face: torch.Tensor) -> torch.Tensor:
    activations, gradients = [], []
    def forward_hook(module, inputs, output):
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))
    handle = model.encoder.features[-1].register_forward_hook(forward_hook)
    try:
        model.eval()
        model.zero_grad()
        with torch.enable_grad():
            model(face).sum().backward()
        cam = (activations[0] * gradients[0].mean((2, 3), keepdim=True)).sum(1, keepdim=True).relu()
        cam = F.interpolate(cam, size=face.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam/(cam.amax((2, 3), keepdim=True) + 1e-8)
        return cam.detach().squeeze(1)
    finally:
        handle.remove()
