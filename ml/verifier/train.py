"""
Verifier training + scoring on cached windows (MPS or CPU).

- masked BCE: each candidate trains only its own class head;
- balanced sampler: `positive_share` of every epoch's draws are positives
  (with replacement), the rest negatives (without replacement when possible);
- augmentation (training only): the same random shift (dx, dy) in
  [-jitter, jitter]^2 for both crops (cut on the CPU from the cached window by
  exactly infer.verifier.crops_from_window's geometry), then on the device one
  D4 element and one set of photometric parameters (brightness, contrast,
  saturation, hue) shared by both crops of a sample;
- AdamW, lr_head for the new layers / lr_backbone for MobileNetV3, cosine
  schedule over a FIXED number of epochs (no early stopping: nothing is
  selected on the folds the model later scores); fixed seeds.

Scoring uses the centre crops through infer.verifier.batch_inputs — the same
preprocessing the ONNX runtime path uses.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np
import torch
from torch import nn

from infer.verifier import LARGE, SMALL, batch_inputs, crops_from_window
from verifier.config import VerifierConfig
from verifier.model import TwoCropVerifier, freeze_batchnorm

logger = logging.getLogger("ml.verifier.train")

GRAY = (0.299, 0.587, 0.114)


def pick_device(name: str = "auto") -> torch.device:
    name = (name or "auto").lower()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "auto":
        return torch.device("mps" if mps_ok else "cpu")
    if name == "mps" and not mps_ok:
        logger.warning("MPS not available — training on CPU")
        return torch.device("cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available — training on CPU")
        return torch.device("cpu")
    return torch.device(name)


# ------------------------------------------------------------------ augmentation


def d4_np(img: np.ndarray, k: int) -> np.ndarray:
    """D4 element k = r + 4f on an HWC image (a view): flip left-right if f,
    then rot90 counter-clockwise r times — infer.tiling's convention."""
    if k // 4:
        img = img[:, ::-1]
    return np.rot90(img, k % 4, axes=(0, 1)) if k % 4 else img


def jittered_crops(windows: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                   ks: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(small [B,64,64,3], large [B,128,128,3]) uint8: infer.verifier.crops_from_window
    per sample (the inference geometry) with shift (dx, dy), then D4 element k."""
    n = len(windows)
    small = np.empty((n, SMALL, SMALL, 3), dtype=np.uint8)
    large = np.empty((n, LARGE, LARGE, 3), dtype=np.uint8)
    for i in range(n):
        s, l = crops_from_window(windows[i], int(dx[i]), int(dy[i]))
        if ks is not None and ks[i]:
            s, l = d4_np(s, int(ks[i])), d4_np(l, int(ks[i]))
        small[i] = s
        large[i] = l
    return small, large


def _to_device(a: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(a).to(device).permute(0, 3, 1, 2).float().div_(255.0)


def _hue_matrices(h: np.ndarray) -> np.ndarray:
    """Rotation by 2*pi*h about the grey axis (1,1,1)/sqrt(3) — a hue shift."""
    th = 2.0 * np.pi * h
    c, s = np.cos(th), np.sin(th)
    k = np.ones(3) / np.sqrt(3.0)
    kk = np.outer(k, k)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    eye = np.eye(3)
    return (c[:, None, None] * eye + (1 - c)[:, None, None] * kk + s[:, None, None] * kx).astype(np.float32)


def _photometric(x: torch.Tensor, b, c, s, hue_m, ref_mean) -> torch.Tensor:
    gray_w = torch.tensor(GRAY, device=x.device).view(1, 3, 1, 1)
    x = x * b.view(-1, 1, 1, 1)
    x = (x - ref_mean) * c.view(-1, 1, 1, 1) + ref_mean
    gray = (x * gray_w).sum(1, keepdim=True)
    x = gray + (x - gray) * s.view(-1, 1, 1, 1)
    x = torch.einsum("bij,bjhw->bihw", hue_m, x)
    return x.clamp_(0.0, 1.0)


def augment_batch(windows: np.ndarray, cfg: VerifierConfig, rng: np.random.Generator,
                  device: torch.device):
    """Training crops for a [B,160,160,3] uint8 window batch -> (small, large)."""
    a = cfg.augment
    n = len(windows)
    dx = rng.integers(-a.jitter, a.jitter + 1, size=n)
    dy = rng.integers(-a.jitter, a.jitter + 1, size=n)
    ks = rng.integers(0, 8, size=n) if a.d4 else None
    small_np, large_np = jittered_crops(windows, dx, dy, ks)
    small, large = _to_device(small_np, device), _to_device(large_np, device)
    f32 = np.float32
    b = torch.from_numpy((1 + rng.uniform(-a.brightness, a.brightness, n)).astype(f32)).to(device)
    c = torch.from_numpy((1 + rng.uniform(-a.contrast, a.contrast, n)).astype(f32)).to(device)
    s = torch.from_numpy((1 + rng.uniform(-a.saturation, a.saturation, n)).astype(f32)).to(device)
    hm = torch.from_numpy(_hue_matrices(rng.uniform(-a.hue, a.hue, n))).to(device)
    gray_w = torch.tensor(GRAY, device=device).view(1, 3, 1, 1)
    # contrast pivots on the LARGE crop's mean grey (after brightness) for both
    # crops, so the small crop stays exactly the centre of the large one
    ref = ((large * b.view(-1, 1, 1, 1)) * gray_w).sum(1, keepdim=True).mean((2, 3), keepdim=True)
    return _photometric(small, b, c, s, hm, ref), _photometric(large, b, c, s, hm, ref)


# ------------------------------------------------------------------ training


def _balanced_order(y: np.ndarray, share: float, rng: np.random.Generator) -> np.ndarray:
    n = len(y)
    pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    if not len(pos) or not len(neg):
        return rng.permutation(n)
    n_pos = max(1, int(round(share * n)))
    n_neg = n - n_pos
    order = np.concatenate([
        rng.choice(pos, n_pos, replace=n_pos > len(pos)),
        rng.choice(neg, n_neg, replace=n_neg > len(neg)),
    ])
    return rng.permutation(order)


def train_model(
    name: str,
    windows: np.ndarray,
    ids: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    num_classes: int,
    cfg: VerifierConfig,
    device: torch.device,
    seed: int,
) -> tuple[TwoCropVerifier, dict]:
    """Train one verifier on the candidates `ids` (labels 0/1, class per row)."""
    t_cfg = cfg.train
    ids = np.asarray(ids, dtype=np.int64)
    y_all = np.asarray(labels, dtype=np.float32)
    c_all = np.asarray(classes, dtype=np.int64)
    if not (len(ids) == len(y_all) == len(c_all)) or not len(ids):
        raise ValueError(f"{name}: empty or misaligned training set")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TwoCropVerifier(
        num_classes, pretrained=cfg.model.pretrained == "imagenet1k_v1",
        embed_dim=cfg.model.embed_dim, dropout=cfg.model.dropout,
    ).to(device)
    opt = torch.optim.AdamW(
        [{"params": model.backbone_parameters(), "lr": t_cfg.lr_backbone},
         {"params": model.head_parameters(), "lr": t_cfg.lr_head}],
        weight_decay=t_cfg.weight_decay,
    )
    steps_per_epoch = math.ceil(len(ids) / t_cfg.batch_size)
    total = max(1, t_cfg.epochs * steps_per_epoch)
    if t_cfg.schedule == "cosine":
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total)))
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    loss_fn = nn.BCEWithLogitsLoss()
    log = {"name": name, "n_train": int(len(ids)), "n_pos": int((y_all == 1).sum()),
           "n_neg": int((y_all == 0).sum()), "epochs": t_cfg.epochs, "seed": seed,
           "device": str(device), "weights_source": model.weights_source,
           "batch_size": t_cfg.batch_size, "epoch_log": []}
    logger.info(f"[{name}] training on {len(ids)} candidates ({log['n_pos']} positive) "
                f"for {t_cfg.epochs} epochs on {device} (backbone {model.weights_source})")
    t_start = time.perf_counter()
    for epoch in range(t_cfg.epochs):
        model.train()
        if t_cfg.freeze_bn:
            freeze_batchnorm(model)
        order = _balanced_order(y_all, t_cfg.positive_share, rng)
        t0, seen = time.perf_counter(), 0
        loss_sum = torch.zeros((), device=device)  # summed on the device: no per-step sync
        for s in range(0, len(order), t_cfg.batch_size):
            batch = order[s:s + t_cfg.batch_size]
            wb = np.asarray(windows[ids[batch]])
            small, large = augment_batch(wb, cfg, rng, device)
            yb = torch.from_numpy(y_all[batch]).to(device)
            cb = torch.from_numpy(c_all[batch]).to(device)
            logits, _ = model(small, large)
            z = logits.gather(1, cb.view(-1, 1)).squeeze(1)
            loss = loss_fn(z, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            seen += len(batch)
            loss_sum += loss.detach() * len(batch)
        epoch_loss = float(loss_sum) / max(seen, 1)  # syncs the device once per epoch
        dt = time.perf_counter() - t0
        entry = {"epoch": epoch + 1, "loss": round(epoch_loss, 5),
                 "seconds": round(dt, 2), "samples_per_s": round(seen / max(dt, 1e-9), 1)}
        log["epoch_log"].append(entry)
        logger.info(f"[{name}] epoch {epoch + 1}/{t_cfg.epochs} loss {entry['loss']:.4f} "
                    f"{entry['seconds']} s ({entry['samples_per_s']} cand/s)")
    log["seconds"] = round(time.perf_counter() - t_start, 2)
    model.eval()
    return model, log


@torch.no_grad()
def predict(
    model: TwoCropVerifier,
    windows: np.ndarray,
    ids: np.ndarray,
    classes: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
    return_embedding: bool = False,
):
    """Verifier logits of each candidate's own class head (centre crops)."""
    model.eval()
    ids = np.asarray(ids, dtype=np.int64)
    classes = np.asarray(classes, dtype=np.int64)
    out, embs = [], []
    for s in range(0, len(ids), batch_size):
        feed = batch_inputs(np.asarray(windows[ids[s:s + batch_size]]))
        small = torch.from_numpy(feed["crop_small"]).to(device)
        large = torch.from_numpy(feed["crop_large"]).to(device)
        logits, emb = model(small, large)
        cb = torch.from_numpy(classes[s:s + batch_size]).to(device)
        out.append(logits.gather(1, cb.view(-1, 1)).squeeze(1).float().cpu().numpy())
        if return_embedding:
            embs.append(emb.float().cpu().numpy())
    z = np.concatenate(out).astype(np.float64) if out else np.zeros(0)
    if return_embedding:
        return z, (np.concatenate(embs) if embs else np.zeros((0, model.fc.out_features)))
    return z
