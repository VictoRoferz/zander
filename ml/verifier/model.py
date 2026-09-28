"""
TwoCropVerifier — the stage-2 network (CONTRACTS §8.4):

    crop_small [N,3,64,64]  --+                          (RGB, 0..1; ImageNet
    crop_large [N,3,128,128] -+-> shared MobileNetV3-Small  normalisation is
                                  features -> global pool   inside the graph)
                                  576 per crop -> concat 1152
                                  -> Linear 1152->128 -> Hardswish = embedding [N,128]
                                  -> Dropout 0.2 -> Linear 128->C = logits [N,C]

One head per schema class; training uses only the candidate's own class head
(masked BCE, train.py). The backbone is torchvision's mobilenet_v3_small with
IMAGENET1K_V1 weights; if they cannot be downloaded the model falls back to a
random init and says so (`weights_source`), which is only acceptable for smoke
runs.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import ssl
import tempfile
import urllib.request
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger("ml.verifier.model")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _download_with_certifi(url: str) -> None:
    """python.org builds on macOS ship without a CA bundle -> urllib fails with
    CERTIFICATE_VERIFY_FAILED. Fetch the checkpoint into torch's hub cache with
    certifi's bundle instead and verify torchvision's sha256 prefix."""
    import certifi

    dest = Path(torch.hub.get_dir()) / "checkpoints" / Path(url).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx, timeout=60) as r, tempfile.NamedTemporaryFile(
        dir=dest.parent, delete=False
    ) as f:
        shutil.copyfileobj(r, f)
        tmp = Path(f.name)
    prefix = Path(url).stem.rsplit("-", 1)[-1]
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if not digest.startswith(prefix):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {url}: {digest[:8]} != {prefix}")
    tmp.replace(dest)


def mobilenet_v3_small_features(pretrained: bool = True) -> tuple[nn.Module, str]:
    """(features module, weights source)."""
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    if not pretrained:
        return mobilenet_v3_small(weights=None).features, "none"
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    try:
        return mobilenet_v3_small(weights=weights).features, "imagenet1k_v1"
    except Exception as first:  # noqa: BLE001 - any download failure falls through
        err: Exception = first
    try:
        _download_with_certifi(weights.url)
        return mobilenet_v3_small(weights=weights).features, "imagenet1k_v1"
    except Exception as second:  # noqa: BLE001
        err = second
    msg = f"none (ImageNet weights unavailable: {type(err).__name__}: {str(err)[:160]})"
    logger.warning("verifier backbone falls back to RANDOM init — smoke runs only: %s", msg)
    return mobilenet_v3_small(weights=None).features, msg


class TwoCropVerifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, embed_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.features, self.weights_source = mobilenet_v3_small_features(pretrained)
        feat_dim = self.features[-1][0].out_channels  # 576 for MobileNetV3-Small
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2 * feat_dim, embed_dim)
        self.act = nn.Hardswish()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def _pooled(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features((x - self.mean) / self.std)).flatten(1)

    def forward(self, crop_small: torch.Tensor, crop_large: torch.Tensor):
        emb = self.act(self.fc(torch.cat([self._pooled(crop_small), self._pooled(crop_large)], 1)))
        return self.head(self.drop(emb)), emb

    def backbone_parameters(self):
        return list(self.features.parameters())

    def head_parameters(self):
        return list(self.fc.parameters()) + list(self.head.parameters())


def freeze_batchnorm(model: nn.Module) -> None:
    """Keep every BatchNorm in eval mode (running stats frozen; affine params
    still train). The shared backbone sees two crop sizes whose batch
    statistics differ; frozen stats keep train and inference identical."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
