"""
DEIM / D-FINE-S runner — the Apache-2.0 challenger to the (AGPL) Ultralytics
models: one run config YAML -> a bundle v2 zip (docs/CONTRACTS.md §8.1),
exactly like train/ultralytics_run.py, so `cli.py import-run / eval /
choose-op / export` treat both frameworks alike. Normally reached through
train/dispatch.py (`framework: deim` in the config). Run from ml/:

    python train/dispatch.py --config configs/matrix/deim_s_def3_fold0.yaml --device 0
    python train/deim_run.py --config configs/matrix/deim_s_def3_fold0.yaml --dry-run   # the plan, no side effects
    python train/deim_run.py --install-deps                     # Colab: DEIM's extra runtime deps (pinned below)
    python train/deim_run.py --fetch-weights [--drive MyDrive/ZanderML]   # COCO-pretrained DEIM-S init

At arm's length: DEIM v1 (github.com/ShihuaHuang95/DEIM, Apache-2.0) is cloned
at DEIM_COMMIT into ml_data/run_dirs/_deim_src and never modified. This module
writes the dataset layout + ONE override YAML and runs DEIM's own train.py and
tools/deployment/export_onnx.py in subprocesses. DEIM's dependencies never
enter ml/requirements*.txt. NOT DEIMv2 / EdgeCrafter: relicensed
non-commercial on 2026-08-24; the LICENSE of the pinned commit is checked
(sha256 + text) at every run and archived in the bundle as LICENSE_DEIM.

Run policy (the effective config is what the bundle records; its sha256 is
the run's identity). Everything below is derived from `epochs` E, the batch
and the size of OUR training view — DEIM's COCO numbers (118k images, total
batch 32) do not transfer:
  - model deim_hgnetv2_s_coco: DEIM-D-FINE-S fine-tuned (`train.py -t`) from
    DEIM's own COCO checkpoint (the model zoo of the pinned README), like the
    COCO-pretrained YOLO weights. Class heads are re-initialised (80 -> n).
  - schedule: no_aug = max(2, round(0.1 E)), stop = E - no_aug,
    flat = policy[1] = stop // 2, transforms policy [4, flat, stop]; LR:
    quadratic warm-up over 2 epochs, flat until `flat`, cosine to 0.5 lr at
    `stop`, then flat. DEIM-S lr 4e-4 (backbone 2e-4) at batch 32, scaled
    linearly with the batch (DEIM README "Customizing Batch Size").
  - EMA: the ramp (`ema.warmups`) spans 2 epochs; the decay keeps DEIM's COCO
    EMA memory of ~2.7 epochs: 1 - 1/(2.7 * iters_per_epoch). Keeping 0.9999
    (a memory of 10,000 iterations = ~48 of our ~208-iteration epochs) would
    put ~40 % of the exported EMA on the weights of the first half of an
    80-epoch run (the flat high-LR phase); with the scaled decay it is < 1e-6.
  - final_weights: last (the only choice). DEIM's own "stage 2" reloads its
    best_stg1.pth checkpoint — selected on the validation view, i.e. the
    fold{k}_val photos that out-of-fold evaluation, choose-op and the
    verifier score later (a selection leak) — whenever the val metric does
    not improve, and last.pth is not even saved in that stage. Setting
    collate_fn.stop_epoch = E switches stage 2 off: last.pth is saved every
    epoch and the trajectory never depends on the val metric. DEIM's
    batch-level multi-scale (480-800 px, the analogue of the Ultralytics
    scale jitter) therefore stays on until the last epoch.
  - augmentation mirrors train/ultralytics_run.py (measured nuisance only):
    horizontal + vertical flips p 0.5 (D4 is baked into the tiles offline),
    ColorJitter brightness 0.35 / saturation 0.4 / hue 0.015 (~ hsv_v/s/h),
    no contrast. No Mosaic (output_size 320 halves 28-px defects and its
    affine rotates them), no RandomZoomOut / RandomIoUCrop (rescale), no
    mixup. DEIM's RandomPhotometricDistort is NOT used: it also permutes the
    RGB channels (p 0.5), and solder colour is signal. torchvision's
    RandomVerticalFlip and ColorJitter are not in DEIM's registry, so the
    train subprocess registers them before it runs DEIM's train.py unchanged
    (`--_worker train`).
  - COCO category ids are rewritten 0..n-1 (= the schema's class index) with
    remap_mscoco_category false, so DEIM's labels == the class index the
    DeimAdapter passes through unchanged (the tiler writes id = index + 1).
  - the override's __include__ is ABSOLUTE: DEIM joins relative includes
    with the override file's own folder.
  - export: the LAST checkpoint is stripped to its EMA weights (`last.pt` in
    the bundle) and exported by DEIM's export_onnx.main (deploy mode, the
    postprocessor's top-300 inside the graph, opset 16, dynamic batch):
    inputs images [N,3,640,640] float /255 + orig_target_sizes [N,2] int64
    (w, h); outputs labels int64 [N,300], boxes float xyxy pixels [N,300,4],
    scores float [N,300] = layout "deim". DEIM's onnxsim step is skipped (no
    CPU gain measured; onnxsim 0.4.x has no wheel for Colab's Python) and the
    TorchScript exporter is forced (dynamo=False), which DEIM targets. A
    5-tile torch-vs-ONNX parity check (tie-tolerant; boxes match by IoU or
    within 0.5 px, because D-FINE also decodes inverted boxes) must pass; its
    numbers go into meta.json.

Resume (Colab disconnects). With `mirror_dir`, the parent process polls
DEIM's output/log.txt (one JSON line per finished epoch) and copies log.txt,
zander_run.json and output/last.pth (as last.pt, weights LAST, torn-write
guarded) to <mirror_dir>/<name>/ at most every ZANDER_MIRROR_INTERVAL_S
seconds plus the final epoch. On start the most advanced usable checkpoint
(local output/last.pth, then the mirror) whose zander_run.json records this
config sha wins: resumable -> DEIM `train.py -r last.pth` (last_epoch, model,
EMA, optimizer and AMP scaler are restored; the LR schedule is a pure function
of the iteration), finished -> export + bundle only, none -> fresh (an
existing run_dir is moved aside, never deleted). A resumed run is
statistically equivalent to an uninterrupted one, not bit-identical (no RNG
state in DEIM checkpoints). Every DEIM subprocess leads its own process
group: stopping a run (SIGINT -> SIGTERM -> SIGKILL, also on a SIGTERM from
matrix.py) stops its DataLoader workers too — orphaned workers otherwise keep
matrix.py's log pipe open forever — and a worker whose runner died takes its
group down instead of training on unobserved.

Environment (runtime knobs: recorded per attempt, NOT part of the config):
  ZANDER_WORKERS=<n>           dataloader workers (train + val)   (dispatch.py --workers)
  ZANDER_CACHE                 ignored (DEIM has no image cache)  (dispatch.py --cache)
  ZANDER_MIRROR_INTERVAL_S     min seconds between mirror copies (default 300; 0 = every epoch)
  ZANDER_ABORT_AFTER_EPOCH=N   TEST HOOK: after epoch N is logged + mirrored, kill the
                               trainer and exit 75 — simulates a Colab disconnect
  ZANDER_DEIM_POLL_S           how often the parent polls log.txt (default 10 s)
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib.util
import json
import logging
import os
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ML_DIR = Path(__file__).resolve().parents[1]
_TRAIN_DIR = Path(__file__).resolve().parent
# Script mode puts ml/train on sys.path[0]; import everything as `train.*` from ml/.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _TRAIN_DIR]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import yaml  # noqa: E402

from config.settings import settings  # noqa: E402
from dataprep.tiler import load_provenance  # noqa: E402
from infer.tiling import TILE_SIZE  # noqa: E402
from train import bundle as B  # noqa: E402
from train import ultralytics_run as U  # noqa: E402  (shared run-state helpers of the reference runner)

logger = logging.getLogger("ml.deim")

FRAMEWORK = "deim"

# ------------------------------------------------------------ pinned upstream

DEIM_REPO = "https://github.com/ShihuaHuang95/DEIM"
# DEIM v1 HEAD, pinned 2026-09-28. NOT DEIMv2 (Intellindust-AI-Lab/DEIMv2) —
# relicensed non-commercial on 2026-08-24.
DEIM_COMMIT = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"
# sha256 of LICENSE at DEIM_COMMIT (Apache-2.0 with an Intellindust copyright
# header, checked 2026-09-28). Changing DEIM_COMMIT means re-reading the
# licence and updating this pin — checked at every run, never from memory.
DEIM_LICENSE_SHA256 = "f25a3c7b894c50f203813bc72183a22124c89c27b15d420b92e4621ca4b0f9cf"
LICENSE_NAME = "Apache-2.0"
SRC_DIR_NAME = "_deim_src"

# Extra runtime deps of DEIM's train/export path on Colab (torch, torchvision,
# numpy, Pillow, PyYAML, scipy, tensorboard, transformers, accelerate,
# huggingface_hub come preinstalled). Pinned: the newest faster-coco-eval
# before the DEIM commit (DEIM reads its private _evalImgs_cpp), onnxruntime =
# the Mac-side pin (the parity check). "if-missing" entries are only installed
# when absent, never pinned, so Colab's stack is not reshuffled. DEIM imports
# calflops at start-up, and calflops' __init__ imports transformers,
# accelerate and huggingface_hub: without them DEIM dies before training.
DEIM_PIP = (
    # (import name, pip requirement, policy)
    ("faster_coco_eval", "faster-coco-eval==1.7.2", "pin"),
    ("calflops", "calflops==0.3.2", "pin"),
    ("onnxruntime", "onnxruntime==1.20.1", "pin"),
    ("onnx", "onnx==1.17.0", "if-missing"),
    ("scipy", "scipy", "if-missing"),
    ("tensorboard", "tensorboard", "if-missing"),
    ("transformers", "transformers", "if-missing"),        # calflops
    ("accelerate", "accelerate", "if-missing"),            # calflops
    ("huggingface_hub", "huggingface_hub", "if-missing"),  # calflops
    ("gdown", "gdown", "if-missing"),  # --fetch-weights only
)
TRAIN_MODULES = ("torch", "torchvision", "faster_coco_eval", "calflops", "transformers", "accelerate",
                 "huggingface_hub", "scipy", "tensorboard", "PIL", "onnx", "onnxruntime")

BASE_CONFIG = "configs/deim_dfine/deim_hgnetv2_s_coco.yml"
MODELS = {
    # DEIM-D-FINE-S fine-tuned from DEIM's own COCO checkpoint (49.0 AP; model
    # zoo row "S" of the README at DEIM_COMMIT; Google Drive is its only host).
    "deim_hgnetv2_s_coco": {
        "base_config": BASE_CONFIG,
        "init": "tuning",
        "weights_file": "deim_hgnetv2_s_coco.pth",
        "gdrive_id": "1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC",
        "source": "https://drive.google.com/file/d/1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC/view",
    },
    # Same network, random init: plumbing smoke runs only (no download; the
    # weights are meaningless).
    "deim_hgnetv2_s_random": {"base_config": BASE_CONFIG, "init": "random"},
}

# DEIM's registry at DEIM_COMMIT (engine/data/transforms) + the two torchvision
# transforms the train worker registers (see EXTRA_TRANSFORMS).
DEIM_TRANSFORMS = frozenset({
    "RandomPhotometricDistort", "RandomZoomOut", "RandomHorizontalFlip", "Resize",
    "SanitizeBoundingBoxes", "RandomCrop", "Normalize", "EmptyTransform", "PadToSize",
    "RandomIoUCrop", "ConvertBoxes", "ConvertPILImage", "Mosaic",
})
EXTRA_TRANSFORMS = ("RandomVerticalFlip", "ColorJitter")

# ------------------------------------------------------------ run policy

DEFAULT_AUGMENT = {
    "hflip": 0.5,        # RandomHorizontalFlip p        (Ultralytics fliplr)
    "vflip": 0.5,        # RandomVerticalFlip p          (flipud; registered by the worker)
    "brightness": 0.35,  # ColorJitter brightness        (~hsv_v)
    "saturation": 0.4,   # ColorJitter saturation        (~hsv_s)
    "hue": 0.015,        # ColorJitter hue               (~hsv_h: solder colour is signal)
    "contrast": 0.0,     # ColorJitter contrast          (not in the Ultralytics policy)
    "multiscale": True,  # DEIM batch-level multi-scale 480..800 px (~ scale jitter)
}
DEFAULTS = {
    "framework": FRAMEWORK,
    "adapter": FRAMEWORK,
    "model": "deim_hgnetv2_s_coco",
    "fold": "fold0",
    "epochs": 80,
    "imgsz": TILE_SIZE,
    "batch": 16,
    "seed": settings.seed,
    "final_weights": "last",
    "augment": DEFAULT_AUGMENT,
    "extra": {},
}
ALLOWED_KEYS = frozenset({
    "name", "framework", "adapter", "model", "dataset_id", "fold", "epochs", "imgsz",
    "batch", "seed", "final_weights", "augment", "extra",
})
# `extra`: smoke-run subsampling of the views only (deterministic by seed).
EXTRA_KEYS = frozenset({"fraction", "val_fraction"})

NO_AUG_FRACTION = 0.1        # last ~10 % of the epochs without photometric aug, LR at its floor
MIN_NO_AUG_EPOCHS = 2
POLICY_START_EPOCH = 4       # DEIM's first stage: no photometric aug for the first 4 epochs
WARMUP_EPOCHS = 2            # LR warm-up and EMA ramp, in epochs of OUR view
# DEIM's COCO recipe: EMA decay 0.9999 = a memory of 1 / (1 - 0.9999) =
# 10,000 iterations = 10,000 / 3,696 (118,287 images at total batch 32)
# ~ 2.7 COCO epochs. We keep the memory in epochs, not in iterations.
EMA_HORIZON_EPOCHS = 2.7
REFERENCE_BATCH = 32         # DEIM-S learning rates are tuned for a total batch of 32
BASE_LR = 0.0004             # configs/deim_dfine/deim_hgnetv2_s_coco.yml @ DEIM_COMMIT
BASE_BACKBONE_LR = 0.0002
LR_GAMMA = 0.5               # min lr = 0.5 * lr (configs/base/deim.yml)
MULTISCALE_REPEAT = 20       # base_size_repeat of dfine_hgnetv2_s_coco.yml (20 x 640, 5 below, 5 above)
NO_PERIODIC_CHECKPOINTS = 10 ** 9   # checkpoint_freq: resume uses last.pth; no checkpointNNNN.pth copies
DEFAULT_WORKERS = 4          # DEIM's dataloader default when ZANDER_WORKERS is unset

# parity check (torch fp32 CPU vs ONNX Runtime CPU on the exported graph)
PARITY_TILES = 5
PARITY_SCORE_FLOOR = 0.01
PARITY_SCORE_TOL = 1e-3
PARITY_MIN_IOU = 0.98
# ... or every coordinate within this many pixels: D-FINE decodes inverted and
# zero-area boxes (x2 < x1, ...), whose IoU is 0 even against their own twin
# 0.02 px away (measured torch vs ONNX: <= 0.08 px on every box).
PARITY_BOX_TOL_PX = 0.5

ENV_POLL = "ZANDER_DEIM_POLL_S"
DEFAULT_POLL_S = 10.0
ABORT_EXIT_CODE = U.ABORT_EXIT_CODE
PARITY_MARKER = "ZANDER_PARITY="
_FOLD_RE = re.compile(r"^fold\d+$")
_GPU_RE = re.compile(r"^(?:cuda:)?(\d+)$")


# ------------------------------------------------------------------ config


def resolve_config(raw: dict, overrides: Optional[dict] = None) -> dict:
    """Effective config = DEFAULTS <- raw <- overrides, validated. Pure (no
    torch, no DEIM, no data), so train/matrix.py can compare config shas
    before it decides to skip a finished run. Raises bundle.ConfigError."""
    eff = B.effective_config(raw, overrides, defaults=DEFAULTS, allowed_keys=ALLOWED_KEYS)
    errors: list[str] = []

    def need(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    def is_int(v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    def is_num(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    need(eff.get("framework") == FRAMEWORK, f"framework must be {FRAMEWORK!r} for this runner")
    need(eff.get("adapter") == FRAMEWORK, f"adapter must be {FRAMEWORK!r} (infer/onnx_adapters.DeimAdapter)")
    need(eff.get("model") in MODELS, f"model must be one of {sorted(MODELS)}, got {eff.get('model')!r}")
    need(isinstance(eff.get("dataset_id"), str) and bool(eff.get("dataset_id")), "`dataset_id` is required")
    fold = str(eff.get("fold"))
    eff["fold"] = fold
    need(fold == "dev_all" or bool(_FOLD_RE.match(fold)), f"fold must be fold<k> or dev_all, got {fold!r}")
    need(is_int(eff.get("epochs")) and eff["epochs"] >= 1, "`epochs` must be an int >= 1")
    need(is_int(eff.get("batch")) and eff["batch"] >= 1, "`batch` must be an int >= 1 (DEIM's total batch)")
    need(is_int(eff.get("seed")), "`seed` must be an int")
    need(eff.get("imgsz") == TILE_SIZE,
         f"imgsz must be {TILE_SIZE}: the tiles, eval_spatial_size and the exported graph are {TILE_SIZE} px")
    need(eff.get("final_weights") == "last",
         "final_weights must be `last` for DEIM: its best_stg*.pth are selected on the fold's "
         "validation view (a selection leak), and last.pth is what the run policy keeps")
    for key in ("augment", "extra"):
        need(isinstance(eff.get(key), dict), f"`{key}` must be a mapping")
    if not errors:
        aug, extra = eff["augment"], eff["extra"]
        unknown = sorted(set(aug) - set(DEFAULT_AUGMENT))
        need(not unknown, f"unknown augment key(s) {unknown}; allowed {sorted(DEFAULT_AUGMENT)}")
        for key in ("hflip", "vflip"):
            need(is_num(aug.get(key)) and 0.0 <= aug[key] <= 1.0, f"augment.{key} must be in [0, 1]")
        for key in ("brightness", "saturation", "contrast"):
            need(is_num(aug.get(key)) and 0.0 <= aug[key] < 1.0, f"augment.{key} must be in [0, 1)")
        need(is_num(aug.get("hue")) and 0.0 <= aug["hue"] <= 0.5, "augment.hue must be in [0, 0.5]")
        need(isinstance(aug.get("multiscale"), bool), "augment.multiscale must be true/false")
        unknown = sorted(set(extra) - EXTRA_KEYS)
        need(not unknown, f"unknown extra key(s) {unknown}; allowed {sorted(EXTRA_KEYS)} "
                          "(DEIM has no pass-through of trainer arguments)")
        for key in sorted(EXTRA_KEYS & set(extra)):
            need(is_num(extra[key]) and 0.0 < extra[key] <= 1.0, f"extra.{key} must be in (0, 1]")
    parsed = B.parse_run_name(eff["name"])
    if parsed is not None and parsed.fold != fold:
        errors.append(f"name {eff['name']!r} says {parsed.fold} but the config trains fold {fold!r}")
    if errors:
        raise B.ConfigError(f"{eff.get('name')}: " + "; ".join(errors))
    if parsed is None:
        logger.warning(
            f"name {eff['name']!r} does not follow <arch>_<schema>_fold<k> / <arch>_<schema>_devall "
            "(docs/CONTRACTS.md §8.1-8.2): the bundle is its own family — fine for smoke runs only"
        )
    if MODELS[eff["model"]]["init"] == "random":
        logger.warning(f"{eff['name']}: model {eff['model']} starts from RANDOM weights — plumbing only")
    return eff


# ----------------------------------------------------------------- schedule


@dataclass(frozen=True)
class Schedule:
    epochs: int
    no_aug_epoch: int
    stop_epoch: int          # photometric aug off and LR at its floor from here (policy[-1])
    flat_epoch: int          # LR flat until here, then cosine (policy[1])
    policy: tuple            # DEIM transforms policy [start, flat, stop]
    iters_per_epoch: int
    warmup_iter: int
    ema_warmups: int
    ema_decay: float
    collate_stop_epoch: int  # == epochs: DEIM's stage 2 never triggers
    lr: float
    backbone_lr: float
    batch: int
    n_train_images: int


def schedule_for(epochs: int, n_train: int, batch: int) -> Schedule:
    """Everything time-like, derived from E, the batch and the view size."""
    if epochs < 1:
        raise B.ConfigError(f"epochs must be >= 1, got {epochs}")
    if n_train < batch:
        raise B.ConfigError(
            f"the training view has {n_train} images < batch {batch}: DEIM drops the last partial "
            "batch, so an epoch would be empty"
        )
    ipe = n_train // batch  # drop_last: true
    no_aug = min(epochs, max(MIN_NO_AUG_EPOCHS, int(NO_AUG_FRACTION * epochs + 0.5)))
    stop = epochs - no_aug
    flat = stop // 2
    start = min(POLICY_START_EPOCH, flat)
    warmup = WARMUP_EPOCHS * ipe
    ema_decay = round(1.0 - 1.0 / (EMA_HORIZON_EPOCHS * ipe), 6)
    scale = batch / REFERENCE_BATCH
    return Schedule(
        epochs=epochs, no_aug_epoch=no_aug, stop_epoch=stop, flat_epoch=flat,
        policy=(start, flat, stop), iters_per_epoch=ipe, warmup_iter=warmup,
        ema_warmups=warmup, ema_decay=ema_decay, collate_stop_epoch=epochs,
        lr=round(BASE_LR * scale, 10), backbone_lr=round(BASE_BACKBONE_LR * scale, 10),
        batch=batch, n_train_images=n_train,
    )


# ---------------------------------------------------------- dataset layout


@dataclass(frozen=True)
class DatasetLayout:
    img_folder: Path
    train_json: Path
    val_json: Path
    train_view: str
    val_view: str
    n_train_images: int
    n_train_boxes: int
    n_val_images: int
    n_val_boxes: int
    category_map: dict       # tiler category_id -> class index
    classes: list


def views_for(fold: str) -> tuple[str, str]:
    """(train view, val view). dev_all's "val" is fold0_val — inside its
    training data, so only a learning-curve monitor (as for Ultralytics)."""
    if fold == "dev_all":
        return "dev_all_train", "fold0_val"
    return f"{fold}_train", f"{fold}_val"


def remap_coco(doc: dict, classes: list, keep_image_ids: Optional[set] = None) -> tuple[dict, dict]:
    """COCO with category ids 0..n-1 = the class index (schema order).

    The tiler writes category_id = class index + 1; DEIM with
    remap_mscoco_category false uses category_id AS the label, so labels 1..n
    would meet an n-way head. The mapping goes by NAME, and the names in id
    order must equal `classes`. Every annotation is remapped; image ids and
    all other fields are kept. `keep_image_ids` subsamples images (smoke)."""
    cats = doc.get("categories") or []
    by_id = {}
    for c in cats:
        if c["id"] in by_id:
            raise B.ConfigError(f"duplicate COCO category id {c['id']}")
        by_id[c["id"]] = c["name"]
    names = [by_id[i] for i in sorted(by_id)]
    if names != list(classes):
        raise B.ConfigError(f"COCO categories {names} (in id order) != dataset classes {list(classes)}")
    mapping = {cid: list(classes).index(name) for cid, name in by_id.items()}
    images = [im for im in doc["images"] if keep_image_ids is None or im["id"] in keep_image_ids]
    image_ids = {im["id"] for im in images}
    all_ids = {im["id"] for im in doc["images"]}
    anns = []
    for a in doc["annotations"]:
        if a["image_id"] not in all_ids:
            raise B.ConfigError(f"annotation {a.get('id')} refers to unknown image {a['image_id']}")
        if a["image_id"] not in image_ids:
            continue
        if a["category_id"] not in mapping:
            raise B.ConfigError(f"annotation {a.get('id')} has unknown category_id {a['category_id']}")
        anns.append(dict(a, category_id=mapping[a["category_id"]]))
    out = {k: v for k, v in doc.items() if k not in ("images", "annotations", "categories")}
    out.update(images=images, annotations=anns,
               categories=[{"id": i, "name": n} for i, n in enumerate(classes)])
    return out, mapping


def subsample_ids(doc: dict, fraction: float, seed: int, view: str, minimum: int = 1) -> Optional[set]:
    """Deterministic image subset for smoke runs (None = keep everything)."""
    if fraction is None or fraction >= 1.0:
        return None
    ids = sorted(im["id"] for im in doc["images"])
    k = min(len(ids), max(minimum, int(round(fraction * len(ids)))))
    return set(random.Random(f"{seed}:{view}").sample(ids, k))


def prepare_dataset_layout(eff: dict, out_dir: Path, classes: list, *, write: bool = True) -> DatasetLayout:
    """The COCO layout DEIM trains on: the tiler's COCO views with category
    ids 0..n-1, written to <run_dir>/dataset/; images stay where the tiler put
    them (absolute img_folder, no copies, no symlinks)."""
    tiles = settings.tiles_dir / eff["dataset_id"]
    img_folder = (tiles / "images").resolve()
    if not img_folder.is_dir():
        raise FileNotFoundError(
            f"{img_folder} missing — build the tiles (`python cli.py tile --schema <schema>`) or "
            "unzip the tile zip (Colab: the data cell), or set ML_DATA_ROOT"
        )
    train_view, val_view = views_for(eff["fold"])
    extra = eff.get("extra") or {}
    docs, counts, mapping = {}, {}, {}
    for role, view, frac in (("train", train_view, extra.get("fraction")),
                             ("val", val_view, extra.get("val_fraction"))):
        src = tiles / "coco" / f"{view}.json"
        if not src.is_file():
            raise FileNotFoundError(f"{src} missing — rebuild the tiles for {eff['dataset_id']}")
        doc = json.loads(src.read_text(encoding="utf-8"))
        keep = subsample_ids(doc, frac, eff["seed"], view,
                             minimum=eff["batch"] if role == "train" else 1)
        docs[role], mapping = remap_coco(doc, classes, keep)
        counts[role] = (len(docs[role]["images"]), len(docs[role]["annotations"]))
        missing = [im["file_name"] for im in docs[role]["images"][:50]
                   if not (img_folder / im["file_name"]).is_file()]
        if missing:
            raise FileNotFoundError(f"{view}: tile images missing in {img_folder}, e.g. {missing[:3]}")
    out_dir = Path(out_dir)
    layout = DatasetLayout(
        img_folder=img_folder, train_json=out_dir / "train.json", val_json=out_dir / "val.json",
        train_view=train_view, val_view=val_view,
        n_train_images=counts["train"][0], n_train_boxes=counts["train"][1],
        n_val_images=counts["val"][0], n_val_boxes=counts["val"][1],
        category_map={str(k): v for k, v in sorted(mapping.items())}, classes=list(classes),
    )
    if write:
        B.atomic_write_text(layout.train_json, json.dumps(docs["train"], ensure_ascii=False))
        B.atomic_write_text(layout.val_json, json.dumps(docs["val"], ensure_ascii=False))
    return layout


# ------------------------------------------------------------ DEIM config


def _merge(dst: dict, src: dict) -> dict:
    """DEIM's merge_dict: nested dicts merge, everything else (lists!) replaces."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = json.loads(json.dumps(v)) if isinstance(v, (dict, list)) else v
    return dst


def load_deim_config(path: Path) -> dict:
    """DEIM's engine.core.yaml_utils.load_config, re-implemented on plain YAML
    (no torch): includes first, in order, then the file's own keys; a relative
    include is joined with the INCLUDING file's folder, an absolute one is used
    as is. Returns the merged config without `__include__`."""
    path = Path(path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg: dict = {}
    for inc in list(doc.get("__include__") or []):
        inc = os.path.expanduser(str(inc))
        inc_path = Path(inc) if os.path.isabs(inc) else path.parent / inc
        _merge(cfg, load_deim_config(inc_path))
    _merge(cfg, {k: v for k, v in doc.items() if k != "__include__"})
    return cfg


def deim_src_dir() -> Path:
    return settings.run_dirs_dir / SRC_DIR_NAME


def build_override(eff: dict, schedule: Schedule, layout: DatasetLayout, *, deim_dir: Path,
                   output_dir: Path, workers: Optional[int] = None, use_amp: bool = True) -> dict:
    """The one YAML DEIM trains with: an ABSOLUTE include of the pinned base
    config + only what the run policy changes (see the module docstring)."""
    aug, imgsz, s = eff["augment"], eff["imgsz"], schedule
    ops, policy_ops = [], []
    jitter = {k: aug[k] for k in ("brightness", "contrast", "saturation", "hue") if aug[k] > 0}
    if jitter:
        ops.append({"type": "ColorJitter", **jitter})
        policy_ops.append("ColorJitter")
    if aug["hflip"] > 0:
        ops.append({"type": "RandomHorizontalFlip", "p": aug["hflip"]})
    if aug["vflip"] > 0:
        ops.append({"type": "RandomVerticalFlip", "p": aug["vflip"]})
    ops += [
        {"type": "Resize", "size": [imgsz, imgsz]},  # identity on 640-px tiles
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
    ]
    workers = DEFAULT_WORKERS if workers is None else int(workers)
    return {
        "__include__": [str((Path(deim_dir) / MODELS[eff["model"]]["base_config"]).resolve())],
        "output_dir": str(output_dir),
        "num_classes": len(layout.classes),
        "remap_mscoco_category": False,
        "eval_spatial_size": [imgsz, imgsz],
        "use_amp": bool(use_amp),
        "epoches": s.epochs,
        "flat_epoch": s.flat_epoch,
        "no_aug_epoch": s.no_aug_epoch,
        "warmup_iter": s.warmup_iter,
        "lrsheduler": "flatcosine",
        "lr_gamma": LR_GAMMA,
        "checkpoint_freq": NO_PERIODIC_CHECKPOINTS,
        "use_ema": True,
        "ema": {"type": "ModelEMA", "decay": s.ema_decay, "warmups": s.ema_warmups, "start": 0},
        "optimizer": {  # the DEIM-S groups (deim_hgnetv2_s_coco.yml), lr scaled to the batch
            "type": "AdamW",
            "params": [
                {"params": "^(?=.*backbone)(?!.*bn).*$", "lr": s.backbone_lr},
                {"params": "^(?=.*(?:norm|bn)).*$", "weight_decay": 0.0},
            ],
            "lr": s.lr,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
        },
        # init comes from `train.py -t <COCO checkpoint>` (or random for smoke
        # runs): never DEIM's ImageNet-backbone download, whose failure path
        # calls exit() — status 0 — without a process group.
        "HGNetv2": {"pretrained": False},
        "train_dataloader": {
            "total_batch_size": s.batch,
            "num_workers": workers,
            "shuffle": True,
            "dataset": {
                "img_folder": str(layout.img_folder),
                "ann_file": str(layout.train_json),
                "transforms": {
                    "ops": ops,
                    "policy": {"name": "stop_epoch", "epoch": list(s.policy), "ops": policy_ops},
                    "mosaic_prob": 0.0,  # no Mosaic op at all (see the module docstring)
                },
            },
            "collate_fn": {
                "type": "BatchImageCollateFunction",
                "base_size": imgsz,
                "base_size_repeat": MULTISCALE_REPEAT if aug["multiscale"] else None,
                "stop_epoch": s.collate_stop_epoch,  # = epochs: stage 2 (val-selected reloads) off
                "ema_restart_decay": s.ema_decay,    # stage 2 only — unused, kept consistent
                "mixup_prob": 0.0,
                "mixup_epochs": [s.policy[0], s.flat_epoch],
            },
        },
        "val_dataloader": {
            "total_batch_size": max(2, 2 * s.batch),
            "num_workers": workers,
            "dataset": {"img_folder": str(layout.img_folder), "ann_file": str(layout.val_json)},
        },
    }


def render_override(override: dict, header: str = "") -> str:
    """Block-style YAML, never line-wrapped (paths contain spaces), ASCII
    header (DEIM opens configs without an explicit encoding)."""
    lines = ["# GENERATED by ml/train/deim_run.py for one run - do not edit; it is rewritten",
             f"# every attempt. DEIM {DEIM_COMMIT[:7]} ({DEIM_REPO})."]
    lines += [f"# {h}" for h in header.splitlines() if h.strip()]
    return "\n".join(lines) + "\n" + yaml.safe_dump(
        override, sort_keys=False, allow_unicode=True, default_flow_style=False, width=1 << 16)


def _key_paths(d: dict, prefix: tuple = ()):
    for k, v in d.items():
        path = prefix + (k,)
        yield path
        if isinstance(v, dict):
            yield from _key_paths(v, path)


def override_key_problems(override: dict, base: dict) -> list[str]:
    """Every key the override sets must already exist in DEIM's resolved base
    config (a typo would be silently ignored by DEIM), and every transform must
    be registered (DEIM's own or EXTRA_TRANSFORMS)."""
    problems = []
    for path in _key_paths({k: v for k, v in override.items() if k != "__include__"}):
        node = base
        for part in path:
            if not isinstance(node, dict) or part not in node:
                problems.append(".".join(map(str, path)))
                break
            node = node[part]
    ops = (((override.get("train_dataloader") or {}).get("dataset") or {}).get("transforms") or {}).get("ops") or []
    known = DEIM_TRANSFORMS | set(EXTRA_TRANSFORMS)
    problems += [f"transform {op.get('type')!r} is not registered" for op in ops if op.get("type") not in known]
    return problems


def check_override_keys(override: dict, deim_dir: Path, model: str) -> None:
    base = load_deim_config(Path(deim_dir) / MODELS[model]["base_config"])
    problems = override_key_problems(override, base)
    if problems:
        raise B.ConfigError(
            f"the DEIM override sets keys DEIM {DEIM_COMMIT[:7]} does not have: {problems} — "
            "fix train/deim_run.build_override (DEIM would silently ignore them)"
        )


# ------------------------------------------------------------ DEIM source


def _git(*args: str, cwd: Optional[Path] = None, timeout: int = 600) -> str:
    out = subprocess.run(["git", *args], cwd=str(cwd) if cwd else None, capture_output=True,
                         text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed ({out.returncode}): {out.stderr.strip()[-500:]}")
    return out.stdout.strip()


def source_state(dest: Path) -> str:
    """ok | missing | wrong-commit | modified | broken."""
    dest = Path(dest)
    if not (dest / ".git").exists():
        return "missing" if not dest.exists() else "broken"
    try:
        if _git("rev-parse", "HEAD", cwd=dest, timeout=60) != DEIM_COMMIT:
            return "wrong-commit"
        if _git("status", "--porcelain", "--untracked-files=no", cwd=dest, timeout=60):
            return "modified"
    except (RuntimeError, OSError, subprocess.TimeoutExpired):
        return "broken"
    return "ok"


def ensure_deim_source(dest: Optional[Path] = None) -> Path:
    """Clone DEIM_REPO at DEIM_COMMIT into ml_data/run_dirs/_deim_src (reused
    when already there and unmodified). The clone goes to a temp folder first,
    so a half-finished clone never looks valid; an unusable old one is moved
    aside, not deleted."""
    dest = Path(dest or deim_src_dir())
    state = source_state(dest)
    if state == "ok":
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp-{U._utc_stamp()}")
    logger.info(f"cloning DEIM {DEIM_COMMIT[:7]} from {DEIM_REPO} ({state}) -> {dest}")
    try:
        _git("clone", "--quiet", DEIM_REPO, str(tmp))
        _git("-c", "advice.detachedHead=false", "checkout", "--quiet", DEIM_COMMIT, cwd=tmp)
        if _git("rev-parse", "HEAD", cwd=tmp) != DEIM_COMMIT:
            raise RuntimeError(f"checkout did not land on {DEIM_COMMIT}")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if dest.exists() or dest.is_symlink():
        logger.warning(f"DEIM source at {dest} is {state}: moved aside -> {U.move_aside(dest)}")
    os.replace(tmp, dest)
    return dest


_NON_COMMERCIAL = re.compile(r"non[\s-]?commercial|cc[\s-]?by[\s-]?nc|creative\s+commons", re.I)


def check_license(deim_dir: Path) -> tuple[Path, str]:
    """(LICENSE path, sha256): Apache-2.0 text AND the pinned sha, or ConfigError."""
    path = Path(deim_dir) / "LICENSE"
    if not path.is_file():
        raise B.ConfigError(f"{path} missing — cannot verify DEIM's licence")
    text = path.read_text(encoding="utf-8", errors="replace")
    sha = B.sha256_file(path)
    if "Apache License" not in text or "Version 2.0" not in text:
        raise B.ConfigError(f"{path} is not the Apache License 2.0 — refusing to train")
    if _NON_COMMERCIAL.search(text):
        raise B.ConfigError(f"{path} mentions non-commercial terms — refusing to train")
    if sha != DEIM_LICENSE_SHA256:
        raise B.ConfigError(
            f"{path} sha256 {sha[:12]} != pinned {DEIM_LICENSE_SHA256[:12]}: the licence changed — "
            "re-read it before updating DEIM_LICENSE_SHA256"
        )
    return path, sha


# ---------------------------------------------------------- deps + weights


def _has_module(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def missing_modules(names=TRAIN_MODULES) -> list[str]:
    return [n for n in names if not _has_module(n)]


def install_deps(dry_run: bool = False) -> int:
    """pip-install DEIM's extra runtime deps into THIS interpreter (Colab)."""
    reqs = [req for mod, req, policy in DEIM_PIP if policy == "pin" or not _has_module(mod)]
    cmd = [sys.executable, "-m", "pip", "install", "-q", *reqs]
    print("$ " + " ".join(cmd), flush=True)
    if dry_run or not reqs:
        return 0
    return subprocess.run(cmd).returncode


def weights_path(model: str) -> Optional[Path]:
    spec = MODELS[model]
    return settings.pretrained_dir / "deim" / spec["weights_file"] if spec.get("weights_file") else None


def validate_checkpoint(path: Path) -> dict:
    """A loadable DEIM checkpoint with EMA or model weights -> summary."""
    import torch

    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise RuntimeError(f"{path}: not a checkpoint dict")
    sd = (state.get("ema") or {}).get("module") if isinstance(state.get("ema"), dict) else None
    sd = sd if sd is not None else state.get("model")
    if not isinstance(sd, dict) or not sd:
        raise RuntimeError(f"{path}: no `ema.module` / `model` state dict")
    n = sum(int(v.numel()) for v in sd.values() if hasattr(v, "numel"))
    return {"tensors": len(sd), "params": n, "has_ema": "ema" in state,
            "last_epoch": state.get("last_epoch")}


def _tofu_sha(file: Path, sha_file: Path) -> str:
    """Trust on first use: the first copy's sha is recorded, later copies must match."""
    sha = B.sha256_file(file)
    known = B.read_sha256_file(sha_file)
    if known is None:
        B.atomic_write_text(sha_file, f"{sha}  {file.name}\n")
    elif known != sha:
        raise RuntimeError(f"{file} sha256 {sha[:12]} != recorded {known[:12]} ({sha_file}) — "
                           "a different init checkpoint; delete the record only if that is intended")
    return sha


def fetch_weights(model: str = "deim_hgnetv2_s_coco", drive: Optional[Path] = None) -> Optional[Path]:
    """Put the init checkpoint into ml_data/pretrained/deim/: from
    <drive>/pretrained/ if present, else downloaded (gdown) and copied there
    for later sessions. Its sha256 is pinned on first use (<file>.sha256)."""
    spec = MODELS[model]
    target = weights_path(model)
    if target is None:
        print(f"{model}: random init, nothing to fetch")
        return None
    on_drive = Path(drive) / "pretrained" / target.name if drive else None
    if not target.is_file():
        if on_drive is not None and on_drive.is_file():
            print(f"copying {on_drive} -> {target}", flush=True)
            B.atomic_copy(on_drive, target)
        else:
            try:
                import gdown
            except ImportError:
                raise RuntimeError("gdown is not installed (`python train/deim_run.py --install-deps`)") from None
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.part")
            print(f"downloading {spec['source']} -> {target}", flush=True)
            got = gdown.download(id=spec["gdrive_id"], output=str(tmp), quiet=False)
            if not got or not tmp.is_file():
                raise RuntimeError(
                    f"download failed (Google Drive quota?). Download {spec['source']} by hand and put it at "
                    f"{on_drive or target} (the same file name), then rerun"
                )
            os.replace(tmp, target)
    info = validate_checkpoint(target)
    sha = _tofu_sha(target, target.with_name(target.name + ".sha256"))
    if on_drive is not None:
        if not on_drive.is_file():
            B.atomic_copy(target, on_drive)
        _tofu_sha(on_drive, on_drive.with_name(on_drive.name + ".sha256"))
    print(f"{model}: {target} sha256 {sha} ({info['params'] / 1e6:.1f} M params, ema={info['has_ema']})")
    return target


# ---------------------------------------------------------------- commands


def device_plan(device: Optional[str]) -> tuple[list, dict, bool]:
    """(train.py args, env additions, cuda?) for dispatch's --device."""
    d = "" if device is None else str(device).strip().lower()
    if d in ("", "auto"):
        return [], {}, True
    if d == "cpu":
        return ["-d", "cpu"], {}, False
    m = _GPU_RE.match(d)
    if m:
        # DEIM's single-process path picks torch.device('cuda') itself; a
        # `-d cuda:0` would reach torch.autocast(device_type='cuda:0') and fail.
        return [], {"CUDA_VISIBLE_DEVICES": m.group(1)}, True
    raise B.ConfigError(f"device {device!r}: the DEIM runner supports cpu or ONE CUDA index (e.g. 0)")


def subprocess_env(extra: Optional[dict] = None) -> dict:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    # DEIM calls torch.load() without weights_only; torch >= 2.6 defaults it to
    # True. The files are our own checkpoints and the pinned COCO init.
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env.update(extra or {})
    return env


def worker_command(kind: str, deim_dir: Path, *args: str) -> list:
    return [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker", kind,
            "--deim-dir", str(deim_dir), *args]


def train_command(deim_dir: Path, override: Path, *, seed: int, device: Optional[str],
                  resume: Optional[Path] = None, tuning: Optional[Path] = None) -> tuple[list, dict]:
    dev_args, env_extra, cuda = device_plan(device)
    args = ["--", "-c", str(override), "--seed", str(seed), *dev_args]
    if cuda:
        args.append("--use-amp")
    if resume is not None:
        args += ["-r", str(resume)]
    elif tuning is not None:
        args += ["-t", str(tuning)]
    return worker_command("train", deim_dir, *args), subprocess_env(env_extra)


def export_command(deim_dir: Path, override: Path, checkpoint: Path, out_dir: Path) -> list:
    return worker_command("export", deim_dir, "--config", str(override),
                          "--checkpoint", str(checkpoint), "--out", str(out_dir))


def parity_command(deim_dir: Path, override: Path, checkpoint: Path, onnx_path: Path,
                   images: list) -> list:
    return worker_command("parity", deim_dir, "--config", str(override), "--checkpoint",
                          str(checkpoint), "--onnx", str(onnx_path), "--images", *map(str, images))


# --------------------------------------------------------------- run state


@dataclass(frozen=True)
class CkptState:
    kind: str                          # missing | corrupt | resumable | finished
    epochs_done: Optional[int] = None
    detail: str = ""


@dataclass(frozen=True)
class StartPlan:
    kind: str                          # fresh | resume | finished
    source: str                        # none | local | mirror
    epochs_done: Optional[int] = None
    moved_aside: Optional[Path] = None
    reason: str = ""


LOCAL_CKPT = Path("output") / "last.pth"
LOCAL_LOG = Path("output") / "log.txt"
# (run_dir relative path, mirror name) in copy order — the weights LAST, so a
# mirror's log.txt is never behind its checkpoint. last.pt is DEIM's full
# training checkpoint under the Drive layout's name (docs/CONTRACTS.md §8.5).
MIRROR_MEMBERS = ((LOCAL_LOG, "log.txt"), (Path(U.SIDECAR), U.SIDECAR), (LOCAL_CKPT, "last.pt"))


def probe_checkpoint(path: Path, total_epochs: int) -> CkptState:
    """Classify a DEIM training checkpoint by its `last_epoch` (0-based)."""
    path = Path(path)
    if not path.is_file():
        return CkptState("missing")
    try:
        import torch

        state = torch.load(str(path), map_location="cpu", weights_only=False)
        last = int(state["last_epoch"])
        if not (state.get("model") or state.get("ema")):
            raise KeyError("no model/ema weights")
    except Exception as e:  # truncated write, foreign file, ...
        return CkptState("corrupt", detail=f"{type(e).__name__}: {e}")
    done = last + 1
    if done < 1:
        return CkptState("corrupt", detail=f"last_epoch {last}")
    return CkptState("finished" if done >= total_epochs else "resumable", done)


def epochs_in_log(path: Path) -> int:
    """Finished epochs according to DEIM's log.txt (max epoch + 1); torn or
    foreign lines are ignored."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    best = -1
    for line in text.splitlines():
        try:
            rec = json.loads(line)
            best = max(best, int(rec["epoch"]))
        except (ValueError, KeyError, TypeError):
            continue
    return best + 1


def truncate_log(path: Path, epochs_done: int) -> int:
    """Drop log lines of epochs >= epochs_done (a log can be one epoch ahead of
    its checkpoint; DEIM appends on resume). Returns the number dropped."""
    path = Path(path)
    if not path.is_file():
        return 0
    keep, dropped = [], 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True):
        try:
            if int(json.loads(line)["epoch"]) >= epochs_done:
                dropped += 1
                continue
        except (ValueError, KeyError, TypeError):
            pass
        keep.append(line)
    if dropped:
        B.atomic_write_text(path, "".join(keep))
    return dropped


def _usable(ckpt: Path, sidecar_dir: Path, config_sha: str, total: int, probe,
            fingerprint: Optional[str]) -> tuple:
    state = probe(ckpt, total)
    if state.kind in ("missing", "corrupt"):
        return None, f"checkpoint {state.kind}" + (f" ({state.detail})" if state.detail else "")
    side = U.read_sidecar(sidecar_dir)
    if side is None:
        return None, f"no {U.SIDECAR} (provenance unknown)"
    if side.get("config_sha256") != config_sha:
        return None, "config changed (config_sha256 differs)"
    # the schedule (iterations per epoch) follows the view size, which the
    # config sha does not cover: never resume across a dataset rebuild
    if fingerprint and side.get("dataset_fingerprint") not in (None, fingerprint):
        return None, "dataset changed (dataset_fingerprint differs)"
    return state, f"{state.kind} checkpoint ({state.epochs_done} epochs)"


def _rank(state: CkptState) -> tuple:
    return (2, 0) if state.kind == "finished" else (1, state.epochs_done or 0)


def restore_from_mirror(mirror_run: Path, run_dir: Path) -> None:
    for rel, name in MIRROR_MEMBERS:
        src = Path(mirror_run) / name
        if src.is_file():
            B.atomic_copy(src, Path(run_dir) / rel)


def plan_start(run_dir: Path, config_sha: str, total_epochs: int, mirror_run: Optional[Path] = None,
               *, probe: Optional[Callable] = None, stamp: Optional[str] = None,
               fingerprint: Optional[str] = None) -> StartPlan:
    """fresh / resume / finished, and make run_dir match the decision (same
    rules as ultralytics_run.prepare_run_dir, DEIM's file layout)."""
    probe = probe or probe_checkpoint  # looked up at call time
    run_dir = Path(run_dir)
    local, local_why = (None, "no run_dir")
    if run_dir.exists():
        local, local_why = _usable(run_dir / LOCAL_CKPT, run_dir, config_sha, total_epochs, probe,
                                   fingerprint)
    mirror, mirror_why = (None, "no mirror")
    if mirror_run is not None and (Path(mirror_run) / "last.pt").is_file():
        mirror, mirror_why = _usable(Path(mirror_run) / "last.pt", Path(mirror_run), config_sha,
                                     total_epochs, probe, fingerprint)
    reason = f"local: {local_why}; mirror: {mirror_why}"
    moved = None
    if mirror is not None and (local is None or _rank(mirror) > _rank(local)):
        if run_dir.exists():
            moved = U.move_aside(run_dir, stamp)
        restore_from_mirror(Path(mirror_run), run_dir)
        state, source = mirror, "mirror"
    elif local is not None:
        state, source = local, "local"
    else:
        if run_dir.exists():
            moved = U.move_aside(run_dir, stamp)
        return StartPlan("fresh", "none", moved_aside=moved, reason=reason)
    truncate_log(run_dir / LOCAL_LOG, state.epochs_done or 0)
    kind = "resume" if state.kind == "resumable" else "finished"
    return StartPlan(kind, source, state.epochs_done, moved, reason)


def copy_if_stable(src: Path, dst: Path, settle_s: float = 2.0) -> bool:
    """Copy a file another process may be writing (torch.save is not atomic):
    skip it while it is fresh, and discard the copy if it changed meanwhile."""
    st1 = src.stat()
    if time.time() - st1.st_mtime < settle_s:
        return False
    B.atomic_copy(src, dst)
    st2 = src.stat()
    if (st1.st_size, st1.st_mtime_ns) != (st2.st_size, st2.st_mtime_ns):
        try:
            dst.unlink()
        except OSError:
            pass
        return False
    return True


class Supervisor:
    """Parent-side epoch bookkeeping for the DEIM subprocess: DEIM has no
    callbacks, so its log.txt (one JSON line per finished epoch, written
    after last.pth) is the epoch clock."""

    def __init__(self, run_dir: Path, mirror_run: Optional[Path], sidecar: dict, attempt: dict,
                 total_epochs: int, start_epoch: int, *, settle_s: float = 2.0):
        self.run_dir = Path(run_dir)
        self.mirror_run = Path(mirror_run) if mirror_run else None
        self.sidecar, self.attempt = sidecar, attempt
        self.total, self.seen = int(total_epochs), int(start_epoch)
        self.t0 = time.time()
        self.interval = U._env_number(U.ENV_MIRROR_INTERVAL, U.DEFAULT_MIRROR_INTERVAL_S, float)
        self.abort_after = U._env_number(U.ENV_ABORT, None, int)
        self.last_mirror: Optional[float] = None
        self.pending_ckpt = False
        self.settle_s = settle_s

    def mirror_now(self) -> None:
        if self.mirror_run is None:
            return
        t = time.monotonic()
        try:
            for rel, name in MIRROR_MEMBERS:
                src = self.run_dir / rel
                if not src.is_file():
                    continue
                if name == "last.pt":
                    self.pending_ckpt = not copy_if_stable(src, self.mirror_run / name, self.settle_s)
                else:
                    B.atomic_copy(src, self.mirror_run / name)
            self.last_mirror = time.monotonic()
            note = " (checkpoint busy: retried next poll)" if self.pending_ckpt else ""
            logger.info(f"mirror: epoch {self.seen}/{self.total} -> {self.mirror_run} "
                        f"({self.last_mirror - t:.1f}s){note}")
        except Exception as e:  # never let Drive trouble kill a GPU run
            logger.warning(f"mirror copy failed (training continues): {type(e).__name__}: {e}")

    def observe(self) -> Optional[int]:
        """Called every poll; returns ABORT_EXIT_CODE when the test hook fires."""
        done = epochs_in_log(self.run_dir / LOCAL_LOG)
        if done <= self.seen:
            if self.pending_ckpt:
                self.mirror_now()
            return None
        self.seen = done
        self.attempt.update(end_epoch=done, seconds=round(time.time() - self.t0, 1), updated=B.utc_now())
        try:
            U.write_sidecar(self.run_dir, self.sidecar)
        except Exception as e:
            logger.warning(f"could not update {U.SIDECAR}: {e}")
        final = done >= self.total
        abort = self.abort_after is not None and done >= self.abort_after and not final
        due = self.last_mirror is None or final or abort or (
            time.monotonic() - self.last_mirror >= self.interval)
        if due:
            if abort or final:  # the checkpoint of this epoch is on disk: no settle wait
                self.settle_s = 0.0
            self.mirror_now()
        return ABORT_EXIT_CODE if abort else None


STOP_GRACE_S = 10.0  # SIGINT -> SIGTERM -> SIGKILL within ~20 s (matrix.py waits 30 s before its SIGKILL)


def spawn_tree(cmd: list, env: dict, cwd: Path, **kwargs) -> subprocess.Popen:
    """Start a DEIM subprocess as the leader of its OWN process group, so the
    whole tree — incl. PyTorch DataLoader workers, which outlive a killed
    trainer blocked on a full result pipe and keep matrix.py's log pipe open —
    can be stopped together."""
    if os.name == "posix":
        kwargs["start_new_session"] = True
    return subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), env=env, **kwargs)


def _signal_tree(proc: subprocess.Popen, sig: int) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, sig)  # pgid == pid: spawn_tree made it a session leader
        elif sig == getattr(signal, "SIGKILL", None):
            proc.kill()
        else:
            proc.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        pass  # nobody left in the group


def stop_tree(proc: subprocess.Popen, grace_s: float = STOP_GRACE_S) -> None:
    """SIGINT first (DEIM unwinds and its DataLoader shuts its workers down),
    then SIGTERM, then SIGKILL — to the whole group; stragglers are killed
    even when the leader already exited."""
    if os.name != "posix":
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        return
    for sig, wait in ((signal.SIGINT, grace_s), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        if proc.poll() is not None:
            break
        _signal_tree(proc, sig)
        try:
            proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            pass
    _signal_tree(proc, signal.SIGKILL)


@contextlib.contextmanager
def sigterm_raises():
    """matrix.py stops a run with SIGTERM (Popen.terminate); Python's default
    action would end this process WITHOUT cleanup and leave the trainer's
    process group running on the GPU. Raise KeyboardInterrupt instead
    (dispatch.py maps it to exit 130)."""
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    old = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, old)


def run_subprocess(cmd: list, env: dict, cwd: Path, sup: Optional[Supervisor] = None,
                   poll_s: Optional[float] = None) -> int:
    """Run a DEIM subprocess (its output goes straight to our stdout/stderr,
    i.e. to matrix.py's tee) while the supervisor watches the epoch clock."""
    poll_s = U._env_number(ENV_POLL, DEFAULT_POLL_S, float) if poll_s is None else poll_s
    logger.info("$ " + " ".join(str(c) for c in cmd))
    sys.stdout.flush()
    sys.stderr.flush()
    with sigterm_raises():
        proc = spawn_tree(cmd, env, cwd)
        try:
            while True:
                rc = proc.poll()
                if sup is not None and sup.observe() == ABORT_EXIT_CODE:
                    logger.warning(f"{U.ENV_ABORT}={sup.abort_after}: simulating a disconnect after epoch "
                                   f"{sup.seen} (trainer stopped, exit {ABORT_EXIT_CODE})")
                    stop_tree(proc)
                    for h in logging.getLogger().handlers:
                        try:
                            h.flush()
                        except Exception:
                            pass
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(ABORT_EXIT_CODE)
                if rc is not None:
                    break
                time.sleep(poll_s)
            _signal_tree(proc, signal.SIGKILL if os.name == "posix" else signal.SIGTERM)  # stragglers
            if sup is not None:
                sup.observe()  # the last epoch may land between two polls
                if sup.pending_ckpt:
                    sup.settle_s = 0.0
                    sup.mirror_now()
            return rc
        except BaseException:
            stop_tree(proc)
            raise


# ------------------------------------------------------ outputs + checks


def metrics_from_log(path: Path) -> dict:
    """metrics.json: DEIM's per-epoch log rows (train losses, lr, COCO bbox
    stats on the val view), de-duplicated by epoch (last row wins)."""
    by_epoch: dict = {}
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            rec = json.loads(line)
            by_epoch[int(rec["epoch"])] = rec
        except (ValueError, KeyError, TypeError):
            continue
    epochs = [by_epoch[k] for k in sorted(by_epoch)]
    return {"source": "deim output/log.txt", "epochs": epochs, "final": epochs[-1] if epochs else {},
            "coco_eval_bbox_keys": ["AP", "AP50", "AP75", "APs", "APm", "APl",
                                    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]}


def _dims(vi) -> list:
    return [d.dim_value if d.HasField("dim_value") else (d.dim_param or None)
            for d in vi.type.tensor_type.shape.dim]


DEIM_INPUTS = (("images", "FLOAT", 4), ("orig_target_sizes", "INT64", 2))
DEIM_OUTPUTS = (("labels", "INT64", 2), ("boxes", "FLOAT", 3), ("scores", "FLOAT", 2))


def deim_onnx_info(path: Path, class_names: list) -> dict:
    """Static facts about a DEIM deploy export (onnx only, no runtime):
    names/dtypes/ranks must be what infer/onnx_adapters.DeimAdapter expects."""
    import ast

    import onnx

    path = Path(path)
    model = onnx.load(str(path))
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), None)
    inits = {i.name for i in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in inits]
    dtype = onnx.TensorProto.DataType.Name

    def spec(vis):
        return [(v.name, dtype(v.type.tensor_type.elem_type), len(v.type.tensor_type.shape.dim)) for v in vis]

    got_in, got_out = spec(inputs), spec(model.graph.output)
    if tuple(got_in) != DEIM_INPUTS or tuple(got_out) != DEIM_OUTPUTS:
        raise RuntimeError(
            f"{path.name}: not a DEIM deploy export — inputs {got_in} (want {list(DEIM_INPUTS)}), "
            f"outputs {got_out} (want {list(DEIM_OUTPUTS)})"
        )
    props = {p.key: p.value for p in model.metadata_props}
    names = None
    if "names" in props:
        try:
            d = ast.literal_eval(props["names"])
            names = [d[k] for k in sorted(d)]
        except (ValueError, SyntaxError, TypeError):
            names = None
    if names is not None and names != list(class_names):
        raise RuntimeError(f"{path.name}: model class names {names} != dataset classes {list(class_names)}")
    return {
        "file": path.name,
        "sha256": B.sha256_file(path),
        "opset": opset,
        "layout": "deim",
        "output_shapes": [_dims(o) for o in model.graph.output],
        "input_shapes": [_dims(i) for i in inputs],
        "output_names": [o.name for o in model.graph.output],
        "bytes": path.stat().st_size,
        "class_names": names,
    }


def stamp_onnx_metadata(path: Path, class_names: list, run_name: str) -> None:
    """Self-describing export (like the Ultralytics ones): class names etc."""
    import onnx

    model = onnx.load(str(path))
    meta = {
        "names": repr({i: n for i, n in enumerate(class_names)}),
        "num_classes": str(len(class_names)),
        "framework": FRAMEWORK,
        "deim_commit": DEIM_COMMIT,
        "run_name": run_name,
        "imgsz": str(TILE_SIZE),
        "license": LICENSE_NAME,
    }
    keep = [p for p in model.metadata_props if p.key not in meta]
    del model.metadata_props[:]
    for p in keep:
        model.metadata_props.add(key=p.key, value=p.value)
    for k, v in meta.items():
        model.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(model)
    B.atomic_write_bytes(path, model.SerializeToString())


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else (1.0 if list(a) == list(b) else 0.0)


def _proper(box) -> bool:
    return box[2] > box[0] and box[3] > box[1]


def compare_detections(ref: tuple, test: tuple, *, score_floor: float = PARITY_SCORE_FLOOR,
                       score_tol: float = PARITY_SCORE_TOL, min_iou: float = PARITY_MIN_IOU,
                       box_tol_px: float = PARITY_BOX_TOL_PX) -> dict:
    """Tie-tolerant parity of two (labels, boxes, scores) top-k outputs: the
    sorted scores must agree, and every reference detection >= score_floor
    needs a partner with the same label, a score within score_tol and IoU >=
    min_iou or every coordinate within box_tol_px (queries with equal scores
    may come out of TopK in any order; an inverted or zero-area box has IoU 0
    with everything, its own twin included). Detections within score_tol of
    either side's k-th score are skipped: a near-tie there can drop a query
    out of one top-k but not the other. `min_iou` reports proper boxes only;
    `degenerate` counts the matched pairs with an improper box."""
    rl, rb, rs = (list(x) for x in ref)
    tl, tb, ts = (list(x) for x in test)
    n = min(len(rs), len(ts))
    s_ref, s_test = sorted(map(float, rs), reverse=True)[:n], sorted(map(float, ts), reverse=True)[:n]
    max_score = max((abs(a - b) for a, b in zip(s_ref, s_test)), default=0.0)
    cutoff = max(s_ref[-1], s_test[-1]) + score_tol if n else float("inf")
    used, ious, mismatches, compared, degenerate, max_box = set(), [], 0, 0, 0, 0.0
    for i in sorted(range(len(rs)), key=lambda k: -float(rs[k])):
        if float(rs[i]) < score_floor or float(rs[i]) <= cutoff:
            break
        compared += 1
        a = [float(v) for v in rb[i]]
        best, best_key, best_iou = None, None, -1.0
        for j in range(len(ts)):
            if j in used or int(tl[j]) != int(rl[i]) or abs(float(ts[j]) - float(rs[i])) > score_tol:
                continue
            b = [float(v) for v in tb[j]]
            iou, dist = _iou(a, b), max(abs(x - y) for x, y in zip(a, b))
            best_iou = max(best_iou, iou)
            key = (iou >= min_iou or dist <= box_tol_px, -dist)  # a partner, then the nearest one
            if best_key is None or key > best_key:
                best, best_key = j, key
        if best is None or not best_key[0]:
            mismatches += 1
            ious.append(max(best_iou, 0.0))
            continue
        used.add(best)
        b = [float(v) for v in tb[best]]
        if _proper(a) and _proper(b):
            ious.append(_iou(a, b))
        else:
            degenerate += 1
        max_box = max(max_box, -best_key[1])
    return {"compared": compared, "mismatches": mismatches, "degenerate": degenerate,
            "max_score_diff": max_score, "min_iou": round(min(ious), 6) if ious else None,
            "max_box_diff_px": max_box, "ok": mismatches == 0 and max_score <= score_tol}


def parity_images(layout: DatasetLayout, k: int = PARITY_TILES) -> list:
    """k validation tiles, positives first (deterministic)."""
    doc = json.loads(Path(layout.val_json).read_text(encoding="utf-8"))
    pos = {a["image_id"] for a in doc["annotations"]}
    ims = sorted(doc["images"], key=lambda im: (im["id"] not in pos, im["id"]))
    return [layout.img_folder / im["file_name"] for im in ims[:k]]


def run_parity(deim_dir: Path, override: Path, checkpoint: Path, onnx_path: Path, images: list,
               cwd: Path, timeout_s: float = 1800.0) -> dict:
    cmd = parity_command(deim_dir, override, checkpoint, onnx_path, images)
    with sigterm_raises():
        proc = spawn_tree(cmd, subprocess_env(), cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except BaseException:
            stop_tree(proc)
            raise
    for line in reversed(stdout.splitlines()):
        if line.startswith(PARITY_MARKER):
            return json.loads(line[len(PARITY_MARKER):])
    raise RuntimeError(f"parity worker failed ({proc.returncode}): {(stderr or stdout)[-2000:]}")


def _versions() -> dict:
    from importlib import metadata

    def ver(dist):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            return None

    versions = {"python": platform.python_version(), "torch": ver("torch"), "cuda": None,
                "torchvision": ver("torchvision"), "onnx": ver("onnx"), "onnxruntime": ver("onnxruntime"),
                "faster_coco_eval": ver("faster-coco-eval"), "numpy": ver("numpy"),
                "deim_commit": DEIM_COMMIT, "deim_repo": DEIM_REPO}
    try:
        import torch

        versions["cuda"] = torch.version.cuda
    except Exception:
        pass
    return versions


# --------------------------------------------------------------------- run


@dataclass
class Prepared:
    """Everything a run (or a dry run) derives before touching the GPU."""
    eff: dict
    config_sha: str
    provenance: dict
    run_dir: Path
    mirror_run: Optional[Path]
    deim_dir: Path
    layout: DatasetLayout
    schedule: Schedule
    override: dict
    override_path: Path
    runtime: dict
    init_path: Optional[Path]


def prepare(config_path: Path, device: Optional[str], overrides: Optional[dict],
            mirror_dir: Optional[Path], *, deim_dir: Path, write: bool) -> Prepared:
    eff = resolve_config(B.load_config(config_path), overrides)
    name, ds_id = eff["name"], eff["dataset_id"]
    provenance = load_provenance(ds_id)
    parsed = B.parse_run_name(name)
    if parsed is not None and provenance.get("schema") != parsed.schema:
        raise B.ConfigError(f"{name}: name says schema {parsed.schema!r} but {ds_id} is "
                            f"{provenance.get('schema')!r}")
    runtime = U.runtime_from_env()
    if runtime.get("cache") is not None:
        logger.info(f"{U.ENV_CACHE}={runtime['cache']!r} ignored: DEIM has no image cache")
    _, _, cuda = device_plan(device)
    run_dir = settings.run_dirs_dir / name
    layout = prepare_dataset_layout(eff, run_dir / "dataset", provenance["classes"], write=write)
    schedule = schedule_for(eff["epochs"], layout.n_train_images, eff["batch"])
    override = build_override(eff, schedule, layout, deim_dir=deim_dir, output_dir=run_dir / "output",
                              workers=runtime.get("workers"), use_amp=cuda)
    return Prepared(
        eff=eff, config_sha=B.config_sha256(eff), provenance=provenance, run_dir=run_dir,
        mirror_run=Path(mirror_dir) / name if mirror_dir else None, deim_dir=Path(deim_dir),
        layout=layout, schedule=schedule, override=override,
        override_path=run_dir / "deim_override.yml", runtime=runtime,
        init_path=weights_path(eff["model"]),
    )


def _override_header(p: Prepared) -> str:
    s = p.schedule
    return (f"run {p.eff['name']} (config sha {p.config_sha[:12]}); view {p.layout.train_view} "
            f"{s.n_train_images} images / batch {s.batch} = {s.iters_per_epoch} it/epoch\n"
            f"policy {list(s.policy)}, no-aug {s.no_aug_epoch}, warm-up {s.warmup_iter} it, "
            f"EMA {s.ema_decay} (ramp {s.ema_warmups} it)")


def run_from_config(config_path: Path, device: Optional[str] = None, overrides: Optional[dict] = None,
                    mirror_dir: Optional[Path] = None) -> Path:
    """Train (or resume / finish) one DEIM config and return its bundle zip
    (docs/CONTRACTS.md §8.1; the dispatch entry point for `framework: deim`)."""
    config_path = Path(config_path)
    overrides = dict(overrides or {})
    resolve_config(B.load_config(config_path), overrides)  # config errors first: pure, no deps
    missing = missing_modules()
    if missing:
        raise B.ConfigError(
            f"DEIM runtime modules missing: {missing} — run `python train/deim_run.py --install-deps` "
            "(the notebook's DEIM cell does it)"
        )
    deim_dir = ensure_deim_source()
    license_path, license_sha = check_license(deim_dir)
    p = prepare(config_path, device, overrides, mirror_dir, deim_dir=deim_dir, write=False)
    eff, name, E = p.eff, p.eff["name"], p.eff["epochs"]
    spec = MODELS[eff["model"]]
    if spec["init"] == "tuning" and not (p.init_path and p.init_path.is_file()):
        raise FileNotFoundError(
            f"init checkpoint {p.init_path} missing — run `python train/deim_run.py --fetch-weights` "
            f"(or download {spec['source']} to that path)"
        )
    check_override_keys(p.override, deim_dir, eff["model"])
    run_root = settings.run_dirs_dir
    run_root.mkdir(parents=True, exist_ok=True)

    fingerprint = p.provenance.get("dataset_fingerprint")
    plan = plan_start(p.run_dir, p.config_sha, E, p.mirror_run, fingerprint=fingerprint)
    logger.info(f"{name}: {plan.kind} (source {plan.source}, epochs done {plan.epochs_done}; {plan.reason})")
    if plan.moved_aside:
        logger.warning(f"previous run_dir moved aside -> {plan.moved_aside}")
    # the dataset layout + override are regenerated every attempt (paths move between VMs)
    prepare_dataset_layout(eff, p.run_dir / "dataset", p.provenance["classes"], write=True)
    B.atomic_write_text(p.override_path, render_override(p.override, _override_header(p)))

    sidecar = U.read_sidecar(p.run_dir) if plan.kind != "fresh" else None
    if sidecar is None:
        sidecar = {"format": 1, "run_name": name, "config_sha256": p.config_sha,
                   "framework": FRAMEWORK, "dataset_fingerprint": fingerprint,
                   "created": B.utc_now(), "attempts": []}
    for a in sidecar["attempts"]:
        if a.get("status") == "running":  # its process is gone
            a["status"] = "interrupted"
    output_dir = p.run_dir / "output"
    if plan.kind in ("fresh", "resume"):
        U._move_stale_bundle(run_root, name)
        resumed = plan.kind == "resume"
        start = (plan.epochs_done or 0) if resumed else 0
        attempt = {
            "n": len(sidecar["attempts"]) + 1, "gpu": U._gpu_name(device), "device": device,
            "started": B.utc_now(), "seconds": 0.0, "resumed": resumed,
            "start_epoch": start, "end_epoch": start, "workers": p.runtime.get("workers"),
            "cache": None, "host": platform.node(), "status": "running",
        }
        sidecar["attempts"].append(attempt)
        output_dir.mkdir(parents=True, exist_ok=True)
        U.write_sidecar(p.run_dir, sidecar)
        cmd, env = train_command(
            deim_dir, p.override_path, seed=eff["seed"], device=device,
            resume=(p.run_dir / LOCAL_CKPT) if resumed else None,
            tuning=p.init_path if (not resumed and spec["init"] == "tuning") else None,
        )
        sup = Supervisor(p.run_dir, p.mirror_run, sidecar, attempt, E, start)
        logger.info(f"training {name}: DEIM {eff['model']} fold={eff['fold']} epochs={E} batch={eff['batch']} "
                    f"{'resume from epoch ' + str(start) if resumed else 'fresh'} runtime={p.runtime}")
        rc = run_subprocess(cmd, env, p.run_dir, sup)
        state = probe_checkpoint(p.run_dir / LOCAL_CKPT, E)
        ok = rc == 0 and state.kind == "finished"
        attempt.update(seconds=round(time.time() - sup.t0, 1), updated=B.utc_now(),
                       status="finished" if ok else "failed",
                       end_epoch=max(sup.seen, state.epochs_done or 0))
        U.write_sidecar(p.run_dir, sidecar)
        if p.mirror_run is not None:  # the mirror's attempt log should agree
            try:
                B.atomic_copy(p.run_dir / U.SIDECAR, p.mirror_run / U.SIDECAR)
            except Exception as e:
                logger.warning(f"mirror sidecar update failed: {e}")
        if not ok:
            raise RuntimeError(
                f"DEIM training {'exited with ' + str(rc) if rc else 'exited 0'} but {LOCAL_CKPT} is "
                f"{state.kind} ({state.epochs_done} of {E} epochs{'; ' + state.detail if state.detail else ''})"
            )
    else:
        logger.info(f"{name}: training already finished — exporting and bundling only")
        U.write_sidecar(p.run_dir, sidecar)

    # ---- export the LAST checkpoint (EMA weights) with DEIM's export code
    export_dir = p.run_dir / "export"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    t_export = time.time()
    rc = run_subprocess(export_command(deim_dir, p.override_path, p.run_dir / LOCAL_CKPT, export_dir),
                        subprocess_env(), p.run_dir)
    onnx_path, weights_pt = export_dir / "model.onnx", export_dir / "last.pth"
    if rc != 0 or not onnx_path.is_file() or not weights_pt.is_file():
        raise RuntimeError(f"DEIM export failed (exit {rc}); see the output above")
    classes = p.provenance["classes"]
    stamp_onnx_metadata(onnx_path, classes, name)
    onnx_info = deim_onnx_info(onnx_path, classes)
    parity = run_parity(deim_dir, p.override_path, weights_pt, onnx_path, parity_images(p.layout), p.run_dir)
    logger.info(f"parity torch vs onnx on {parity.get('tiles')} val tiles: {parity}")
    if not parity.get("ok"):
        raise RuntimeError(f"ONNX export does not match the torch model: {parity}")
    export_seconds = round(time.time() - t_export, 1)

    metrics = metrics_from_log(p.run_dir / LOCAL_LOG)
    epochs_completed = max(epochs_in_log(p.run_dir / LOCAL_LOG),
                           max((a.get("end_epoch") or 0 for a in sidecar["attempts"]), default=0)) or None
    s = p.schedule
    meta = B.build_meta(
        config=eff, provenance=p.provenance, framework=FRAMEWORK, adapter=FRAMEWORK, weights="last",
        epochs_completed=epochs_completed, attempts=sidecar["attempts"], onnx=onnx_info, onnx_raw=None,
        versions=_versions(), overrides=B.to_jsonable(overrides), git=U._git_meta(),
        extra={
            "final_weights": "last",
            "runtime": B.to_jsonable(p.runtime),
            "export_seconds": export_seconds,
            "toolchain": f"{settings.toolchain_name} {settings.toolchain_version}",
            "deim": {"repo": DEIM_REPO, "commit": DEIM_COMMIT, "base_config": spec["base_config"],
                     "license": LICENSE_NAME, "license_sha256": license_sha,
                     "extra_transforms": list(EXTRA_TRANSFORMS), "stage2": "off (collate stop_epoch = epochs)"},
            "init_weights": {"model": eff["model"], "init": spec["init"], "source": spec.get("source"),
                             "file": p.init_path.name if p.init_path else None,
                             "sha256": B.sha256_file(p.init_path) if p.init_path and p.init_path.is_file() else None},
            "schedule": B.to_jsonable(dataclasses.asdict(s)),
            "dataset_view": {k: v for k, v in B.to_jsonable(dataclasses.asdict(p.layout)).items()
                             if k not in ("img_folder", "train_json", "val_json")},
            "parity": parity,
        },
    )
    files = {
        "config.yaml": B.dump_config(eff),
        "meta.json": json.dumps(meta, indent=1, ensure_ascii=False),
        "metrics.json": json.dumps(metrics, indent=1),
        "pip-freeze.txt": U._pip_freeze(),
        "model.onnx": onnx_path,
        "last.pt": weights_pt,  # DEIM checkpoint stripped to its EMA weights (+ last_epoch)
        "LICENSE_DEIM": license_path,
        "args.yaml": yaml.safe_dump(load_deim_config(p.override_path), sort_keys=True, allow_unicode=True),
        "deim_override.yml": p.override_path,
    }
    zip_path, sha = B.write_bundle(run_root / f"{name}_bundle.zip", files)
    logger.info(
        f"bundle ready: {zip_path} (sha256 {sha[:12]}, {meta['epochs_completed']} epochs, "
        f"{meta['train_seconds'] / 60:.1f} train-min over {len(meta['attempts'])} attempt(s)) "
        "— import with `python cli.py import-run`"
    )
    return zip_path


# ----------------------------------------------------------------- dry run


def dry_run(config_path: Path, device: Optional[str] = None, overrides: Optional[dict] = None,
            mirror_dir: Optional[Path] = None) -> str:
    """The full plan without side effects: no clone, no files, no GPU."""
    deim_dir = deim_src_dir()
    p = prepare(Path(config_path), device, overrides, mirror_dir, deim_dir=deim_dir, write=False)
    eff, s, lay = p.eff, p.schedule, p.layout
    spec = MODELS[eff["model"]]
    state = source_state(deim_dir)
    out = [f"# DEIM plan for {eff['name']} (family {B.family_of(eff['name'])}, config sha {p.config_sha[:12]})",
           "", "## effective config", B.dump_config(eff).rstrip(), "",
           "## DEIM source",
           f"repo      {DEIM_REPO}", f"commit    {DEIM_COMMIT}",
           f"clone to  {deim_dir}  (currently: {state})",
           f"licence   {LICENSE_NAME}, LICENSE sha256 pinned {DEIM_LICENSE_SHA256[:16]}…"]
    if state == "ok":
        try:
            check_license(deim_dir)
            out.append("          licence check: ok")
        except B.ConfigError as e:
            out.append(f"          licence check: FAILED — {e}")
    out += ["", "## dataset layout",
            f"images    {lay.img_folder}",
            f"train     {lay.train_view}: {lay.n_train_images} tiles, {lay.n_train_boxes} boxes -> {lay.train_json}",
            f"val       {lay.val_view}: {lay.n_val_images} tiles, {lay.n_val_boxes} boxes -> {lay.val_json}"
            + ("  (monitor only: inside dev_all_train)" if eff["fold"] == "dev_all" else ""),
            f"classes   {lay.classes}; COCO category_id -> label {lay.category_map}",
            "", "## schedule (derived from epochs, batch and the view size)",
            f"epochs {s.epochs}: no-aug {s.no_aug_epoch} (from epoch {s.stop_epoch}), flat {s.flat_epoch}, "
            f"policy {list(s.policy)}, collate stop_epoch {s.collate_stop_epoch} (stage 2 off)",
            f"{s.iters_per_epoch} it/epoch (batch {s.batch}); warm-up {s.warmup_iter} it; EMA decay "
            f"{s.ema_decay}, ramp {s.ema_warmups} it; lr {s.lr} (backbone {s.backbone_lr})",
            "", "## init weights",
            f"{eff['model']} ({spec['init']}): "
            + (f"{p.init_path} ({'present' if p.init_path.is_file() else 'MISSING — --fetch-weights'})"
               if p.init_path else "random init (plumbing only)"),
            "", f"## override ({p.override_path})", render_override(p.override, _override_header(p)).rstrip()]
    if state == "ok":
        problems = override_key_problems(p.override, load_deim_config(deim_dir / spec["base_config"]))
        out.append(f"# key check vs DEIM {DEIM_COMMIT[:7]}: " + ("ok" if not problems else f"UNKNOWN {problems}"))
    else:
        out.append("# key check vs DEIM: skipped (no clone yet; runs before training)")
    fresh, _ = train_command(deim_dir, p.override_path, seed=eff["seed"], device=device,
                             tuning=p.init_path if spec["init"] == "tuning" else None)
    resume, _ = train_command(deim_dir, p.override_path, seed=eff["seed"], device=device,
                              resume=p.run_dir / LOCAL_CKPT)
    exp = export_command(deim_dir, p.override_path, p.run_dir / LOCAL_CKPT, p.run_dir / "export")
    out += ["", "## commands (cwd = run dir)",
            "fresh   $ " + " ".join(map(str, fresh)),
            "resume  $ " + " ".join(map(str, resume)),
            "export  $ " + " ".join(map(str, exp)),
            f"        -> strip {LOCAL_CKPT} to its EMA weights, DEIM tools/deployment/export_onnx.main "
            "(check on, onnxsim off, dynamo off) -> model.onnx, layout deim",
            f"parity  {PARITY_TILES} val tiles, torch fp32 vs onnxruntime (tie-tolerant)",
            f"mirror  {p.mirror_run or '(none: no --mirror-dir)'}: log.txt, {U.SIDECAR}, last.pt",
            f"bundle  {settings.run_dirs_dir / (eff['name'] + '_bundle.zip')}"]
    missing = missing_modules()
    out.append(f"deps    {'ok' if not missing else 'missing ' + str(missing) + ' (--install-deps)'}")
    return "\n".join(out)


# ----------------------------------------------------------------- workers
#
# The DEIM subprocesses run THIS file with --_worker; they import DEIM's
# `engine` from the pinned clone and nothing from ml/ (ml/ is dropped from
# sys.path first, so DEIM and its dependencies can never import our modules).


def watch_parent(poll_s: float = 5.0) -> Optional[threading.Thread]:
    """A worker leads its own process group (spawn_tree). If the runner dies
    (Colab kill, OOM, SIGKILL from matrix.py), take the whole group down —
    DataLoader workers included — instead of training on unobserved."""
    if os.name != "posix" or os.getsid(0) != os.getpid():
        return None  # not started by spawn_tree (e.g. by hand): nothing to guard
    ppid = os.getppid()

    def run() -> None:
        while True:
            time.sleep(poll_s)
            if os.getppid() != ppid:
                os.killpg(os.getpgrp(), signal.SIGKILL)

    thread = threading.Thread(target=run, name="zander-parent-watch", daemon=True)
    thread.start()
    return thread


def _enter_deim(deim_dir: str) -> None:
    sys.path[:] = [p for p in sys.path
                   if Path(p or ".").resolve() not in (ML_DIR.resolve(), _TRAIN_DIR.resolve())]
    sys.path.insert(0, str(Path(deim_dir).resolve()))


def _register_extra_transforms() -> None:
    import torchvision.transforms.v2 as T
    from engine.core import GLOBAL_CONFIG, register

    for name in EXTRA_TRANSFORMS:
        if name not in GLOBAL_CONFIG:
            register()(getattr(T, name))


def _force_torchscript_onnx_exporter() -> None:
    """DEIM's export passes opset 16 + dynamic_axes to torch.onnx.export and
    targets the TorchScript exporter; newer torch defaults to dynamo=True."""
    import functools
    import inspect

    import torch

    original = torch.onnx.export
    if "dynamo" not in inspect.signature(original).parameters:
        return

    @functools.wraps(original)
    def export(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        return original(*args, **kwargs)

    torch.onnx.export = export


def _worker_train(deim_dir: str, train_args: list) -> int:
    _enter_deim(deim_dir)
    _register_extra_transforms()
    import runpy

    script = str(Path(deim_dir) / "train.py")
    sys.argv = [script, *train_args]
    runpy.run_path(script, run_name="__main__")
    return 0


def _load_state(path: str) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _worker_export(deim_dir: str, config: str, checkpoint: str, out: str) -> int:
    _enter_deim(deim_dir)
    _register_extra_transforms()
    _force_torchscript_onnx_exporter()
    import argparse as _ap

    import torch

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(checkpoint)
    ema = state.get("ema")
    stripped = {"last_epoch": state.get("last_epoch"), "date": state.get("date"),
                "zander": {"deim_commit": DEIM_COMMIT, "source": Path(checkpoint).name,
                           "weights": "ema" if ema else "model"}}
    if ema:
        stripped["ema"] = {"module": ema["module"], "updates": ema.get("updates")}
    else:
        stripped["model"] = state["model"]
    stripped_path = out_dir / "last.pth"
    torch.save(stripped, str(stripped_path))
    spec = importlib.util.spec_from_file_location(
        "deim_export_onnx", str(Path(deim_dir) / "tools" / "deployment" / "export_onnx.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # DEIM's own main(): YAMLConfig(config, resume=ckpt) -> load EMA weights ->
    # model.deploy() + postprocessor.deploy() (top-300) -> onnx (opset 16) -> checker.
    # Autograd off: main() runs a batch-32 640-px forward and traces at batch
    # 32; with autograd on, both keep every activation for a backward that never
    # comes (measured peak 32.5 GB vs 3.5 GB, byte-identical ONNX) and a Colab
    # VM without High-RAM OOM-kills the export after the whole training.
    with torch.no_grad():
        module.main(_ap.Namespace(config=config, resume=str(stripped_path), check=True, simplify=False))
    produced = out_dir / "last.onnx"
    if not produced.is_file():
        raise RuntimeError(f"DEIM export wrote no {produced}")
    os.replace(produced, out_dir / "model.onnx")
    return 0


def _worker_parity(deim_dir: str, config: str, checkpoint: str, onnx_path: str, images: list) -> int:
    _enter_deim(deim_dir)
    _register_extra_transforms()
    import numpy as np
    import onnxruntime as ort
    import torch
    from engine.core import YAMLConfig
    from PIL import Image

    cfg = YAMLConfig(config)
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    state = _load_state(checkpoint)
    weights = state["ema"]["module"] if state.get("ema") else state["model"]
    cfg.model.load_state_dict(weights)
    model, post = cfg.model.deploy().eval(), cfg.postprocessor.deploy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rows = []
    for img in images:
        with Image.open(img) as im:
            rgb = np.asarray(im.convert("RGB"))
        x = np.ascontiguousarray((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        size = np.array([[rgb.shape[1], rgb.shape[0]]], dtype=np.int64)
        with torch.no_grad():
            tl, tb, ts = post(model(torch.from_numpy(x)), torch.from_numpy(size))
        ol, ob, os_ = sess.run(["labels", "boxes", "scores"], {"images": x, "orig_target_sizes": size})
        res = compare_detections((tl[0].numpy(), tb[0].numpy(), ts[0].numpy()), (ol[0], ob[0], os_[0]))
        res["file"] = Path(img).name
        rows.append(res)
    summary = {
        "tiles": len(rows), "reference": "torch fp32 CPU, EMA weights, deploy mode",
        "score_floor": PARITY_SCORE_FLOOR, "score_tol": PARITY_SCORE_TOL, "min_iou_required": PARITY_MIN_IOU,
        "box_tol_px": PARITY_BOX_TOL_PX,
        "compared": sum(r["compared"] for r in rows), "mismatches": sum(r["mismatches"] for r in rows),
        "degenerate": sum(r["degenerate"] for r in rows),
        "max_score_diff": max((r["max_score_diff"] for r in rows), default=0.0),
        "max_box_diff_px": max((r["max_box_diff_px"] for r in rows), default=0.0),
        "min_iou": min((r["min_iou"] for r in rows if r["min_iou"] is not None), default=None),
        "files": [r["file"] for r in rows],
    }
    summary["ok"] = bool(rows) and all(r["ok"] for r in rows)
    print(PARITY_MARKER + json.dumps(summary), flush=True)
    return 0


def _worker_main(argv: list) -> int:
    p = argparse.ArgumentParser(prog="deim_run.py --_worker")
    p.add_argument("--_worker", required=True, choices=["train", "export", "parity"])
    p.add_argument("--deim-dir", required=True)
    p.add_argument("--config")
    p.add_argument("--checkpoint")
    p.add_argument("--out")
    p.add_argument("--onnx")
    p.add_argument("--images", nargs="*", default=[])
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    else:
        rest = []
    a = p.parse_args(argv)
    watch_parent()
    if a._worker == "train":
        return _worker_train(a.deim_dir, rest)
    if a._worker == "export":
        return _worker_export(a.deim_dir, a.config, a.checkpoint, a.out)
    return _worker_parity(a.deim_dir, a.config, a.checkpoint, a.onnx, a.images)


# --------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train/deim_run.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="DEIM run config (configs/matrix/deim_*.yaml, configs/final/deim_*.yaml)")
    p.add_argument("--device", default=None, help="cpu or one CUDA index, e.g. 0 (default: cuda if available)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted config override, repeatable (recorded in the bundle)")
    p.add_argument("--mirror-dir", default=None, help="resume mirror root; <dir>/<name>/ per run")
    p.add_argument("--dry-run", action="store_true", help="print the full plan; no clone, no files, no GPU")
    p.add_argument("--install-deps", action="store_true", help="pip-install DEIM's extra runtime deps (Colab)")
    p.add_argument("--check-deps", action="store_true", help="list missing DEIM runtime modules (exit 1 if any)")
    p.add_argument("--fetch-weights", action="store_true", help="put the init checkpoint into ml_data/pretrained/deim/")
    p.add_argument("--model", default="deim_hgnetv2_s_coco", choices=sorted(MODELS), help="for --fetch-weights")
    p.add_argument("--drive", default=None, help="for --fetch-weights: MyDrive/ZanderML (copy from / to pretrained/)")
    return p


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--_worker"]:
        return _worker_main(argv)
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=settings.log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if args.install_deps:
        return install_deps()
    if args.check_deps:
        missing = missing_modules()
        print("DEIM runtime modules: " + ("all present" if not missing else f"missing {missing}"))
        return 1 if missing else 0
    if args.fetch_weights:
        fetch_weights(args.model, Path(args.drive) if args.drive else None)
        return 0
    if not args.config:
        build_parser().error("--config is required (or --install-deps / --check-deps / --fetch-weights)")
    config = Path(args.config)
    if not config.is_file() and (ML_DIR / config).is_file():
        config = ML_DIR / config
    mirror = Path(args.mirror_dir) if args.mirror_dir else None
    overrides = B.parse_overrides(args.overrides)
    if args.dry_run:
        print(dry_run(config, args.device, overrides, mirror))
        return 0
    from train import dispatch  # same exit-code contract as dispatch.py

    return dispatch.main(["--config", str(config)] + (["--device", args.device] if args.device else [])
                         + [f"--set={s}" for s in args.overrides]
                         + (["--mirror-dir", str(mirror)] if mirror else []))


if __name__ == "__main__":
    sys.exit(main())
