"""
Ultralytics training runner: one run config YAML -> a bundle v2 zip
(docs/CONTRACTS.md §8.1). Normally reached through train/dispatch.py:

    python train/dispatch.py --config configs/matrix/yolo26n_def3_fold0.yaml --device 0
    python train/dispatch.py --config configs/smoke_def3.yaml --device cpu   # CPU smoke
    python cli.py train --config ...                                         # same, via the CLI

Output: ml_data/run_dirs/<name>_bundle.zip + <name>_bundle.zip.sha256 — a flat
zip with config.yaml (the EFFECTIVE config), model.onnx (+ model_raw.onnx for
end-to-end models), last.pt (or best.pt), metrics.json (per epoch),
pip-freeze.txt, args.yaml and meta.json. MLflow never syncs across machines;
bundles do (Drive), and `python cli.py import-run <bundle>` registers one.

Run policy — defaults applied to every config (the effective config is what
the bundle records, and its sha256 is the run's identity):
  - final_weights: last. best.pt is selected on fold{i}_val — the photos the
    out-of-fold evaluation, choose-op and the verifier later score (a selection
    leak) — so every run bundles its LAST epoch. dev_all refuses `best`
    outright: its "val" (fold0_val) is inside dev_all_train.
  - extra.patience: 0 — early stopping off (0 = never stop in Ultralytics).
    Early stopping picks the final epoch on fold{i}_val, the same leak as
    `best`: a fold run with patience != 0 gets a loud warning (the
    out-of-fold eval refuses such bundles), dev_all refuses it outright.
  - batch: Ultralytics silently halves it after a CUDA OOM in the first epoch
    of a process; meta.json records what last.pt says it trained with as
    `batch_effective`, and a mismatch is logged as a loud WARNING.
  - deterministic: true (configurable).
  - export: {opset: 17, raw_variant: auto}. model.onnx is the default export
    (YOLO26: the NMS-free one-to-one head (1,300,6) = layout "e2e"; YOLO11: the
    raw head (1,4+nc,N) = "raw"); for end-to-end models `auto` also writes
    model_raw.onnx (end2end=False). `off` skips it.

Resume (Colab disconnects). With `mirror_dir`, an `on_model_save` callback
copies results.csv, args.yaml, last.pt and zander_run.json (the attempt log)
to `<mirror_dir>/<name>/` — each atomically, the sidecar LAST (a VM dying
mid-copy leaves the old sidecar, which then refuses the new checkpoint if it
belongs to another config or dataset — never a new sidecar vouching for an
old checkpoint), at most every ZANDER_MIRROR_INTERVAL_S seconds plus the
final epoch; failures are logged, never fatal. On start the runner picks the
most advanced usable checkpoint (local run_dir first, then the mirror): its
zander_run.json must record the effective config's sha AND today's
dataset_fingerprint, and the checkpoint's own train_args (epochs, imgsz,
seed, batch — or a batch Ultralytics halved after an OOM) must agree with
the config, because a resume trains with the checkpoint's arguments, not ours:
  - resumable -> YOLO(last).train(resume=True, data=<rewritten yaml>, ...)
  - finished  -> straight to export + bundle
  - none      -> fresh start; an existing run_dir is moved aside to
                 `<name>.stale-<UTC stamp>` first (never deleted, never merged:
                 Ultralytics appends to results.csv), and so is an unusable
                 mirror (`<mirror_dir>/<name>.stale-<UTC stamp>`).
An I/O error while reading a candidate (Drive) fails the attempt (retried)
instead of discarding the candidate. A resumed run is statistically
equivalent to an uninterrupted one, not bit-identical (dataloader RNG state
is not checkpointed).

Environment (runtime knobs: recorded per attempt, NOT part of the config):
  ZANDER_WORKERS=<n>           dataloader workers          (dispatch.py --workers)
  ZANDER_CACHE=disk|ram|false  Ultralytics image cache     (dispatch.py --cache)
  ZANDER_MIRROR_INTERVAL_S     min seconds between mirror copies (default 300; 0 = every epoch)
  ZANDER_ABORT_AFTER_EPOCH=N   TEST HOOK: hard-exit (code 75) right after epoch N
                               is saved + mirrored — simulates a Colab disconnect

Augmentation policy v2 (2026-09-28; augment only against MEASURED nuisance
variation — see dataprep/tiler.py for the offline part):
  - orientation: offline D4 per tile (exact on boxes) + online flips here;
    free-angle `degrees` is OFF — a 10-degree rotation inflates a 28-px box's
    enclosing box by ~18 %, i.e. it corrupts the labels of tiny defects.
  - scale jitter capped at 0.2 (fixed camera distance; 0.5 would halve boxes).
  - HSV: brightness spans 122-177 (mean luma) across sessions, so value/
    saturation jitter is moderate; hue stays small (solder colour is signal);
    the large A<->B colour shift is handled by offline colour transfer.
  - mosaic 1.0 + close_mosaic 10 (mixes board-A and board-B tiles in one
    image — may weaken the product shortcut; ablation candidate), no mixup.
  - class balance: repeat-factor sampling lives in the dataset view.
"""
from __future__ import annotations

import ast
import csv
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from config.settings import settings
from dataprep.tiler import load_provenance
from infer.tiling import TILE_SIZE
from train import bundle as B

logger = logging.getLogger("ml.train")

FRAMEWORK = "ultralytics"

DEFAULT_AUGMENT = {
    "fliplr": 0.5,
    "flipud": 0.5,
    "degrees": 0.0,
    "scale": 0.2,
    "translate": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.4,
    "hsv_v": 0.35,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "mixup": 0.0,
    "copy_paste": 0.0,
}

DEFAULTS = {
    "framework": FRAMEWORK,
    "adapter": FRAMEWORK,
    "fold": "fold0",
    "epochs": 120,
    "imgsz": 640,
    "batch": 16,
    "seed": settings.seed,
    "deterministic": True,
    "final_weights": "last",
    "augment": DEFAULT_AUGMENT,
    "extra": {"patience": 0},
    "export": {"opset": 17, "raw_variant": "auto"},
}
ALLOWED_KEYS = B.DEFAULT_ALLOWED_KEYS
EXPORT_KEYS = frozenset({"opset", "raw_variant"})
RAW_VARIANTS = ("auto", "off")
# model.train() arguments the runner owns; under `extra` they would be ambiguous.
RESERVED_EXTRA = frozenset(
    {
        "data", "model", "project", "name", "exist_ok", "resume", "save_dir",
        "epochs", "imgsz", "batch", "seed", "deterministic", "device",
    }
)

SIDECAR = "zander_run.json"
ENV_WORKERS = "ZANDER_WORKERS"
ENV_CACHE = "ZANDER_CACHE"
ENV_MIRROR_INTERVAL = "ZANDER_MIRROR_INTERVAL_S"
ENV_ABORT = "ZANDER_ABORT_AFTER_EPOCH"
DEFAULT_MIRROR_INTERVAL_S = 300.0
ABORT_EXIT_CODE = 75

_FOLD_RE = re.compile(r"^fold\d+$")


# ------------------------------------------------------------------ config


def resolve_config(raw: dict, overrides: Optional[dict] = None) -> dict:
    """Effective config = DEFAULTS <- raw <- overrides, validated.

    Raises train.bundle.ConfigError on anything a retry can't fix."""
    eff = B.effective_config(raw, overrides, defaults=DEFAULTS, allowed_keys=ALLOWED_KEYS)
    errors: list[str] = []

    def need(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    for key in ("model", "dataset_id"):
        need(isinstance(eff.get(key), str) and bool(eff.get(key)), f"`{key}` is required")
    need(eff.get("framework") == FRAMEWORK, f"framework must be {FRAMEWORK!r} for this runner")
    need(eff.get("adapter") == FRAMEWORK, f"adapter must be {FRAMEWORK!r} for this runner")
    fold = str(eff.get("fold"))
    eff["fold"] = fold
    need(fold == "dev_all" or bool(_FOLD_RE.match(fold)), f"fold must be fold<k> or dev_all, got {fold!r}")
    for key, lo in (("epochs", 1), ("imgsz", 32)):
        v = eff.get(key)
        need(isinstance(v, int) and not isinstance(v, bool) and v >= lo, f"`{key}` must be an int >= {lo}")
    need(isinstance(eff.get("seed"), int) and not isinstance(eff.get("seed"), bool), "`seed` must be an int")
    batch = eff.get("batch")
    need(isinstance(batch, (int, float)) and not isinstance(batch, bool), "`batch` must be a number")
    need(isinstance(eff.get("deterministic"), bool), "`deterministic` must be true/false")
    weights = eff.get("final_weights")
    need(weights in B.WEIGHTS_CHOICES, f"final_weights must be one of {B.WEIGHTS_CHOICES}")
    if fold == "dev_all" and weights == "best":
        errors.append(
            "dev_all refuses final_weights: best — its validation view (fold0_val) is inside "
            "dev_all_train, so best.pt would be selected on training data; use last"
        )
    patience = eff["extra"].get("patience") if isinstance(eff.get("extra"), dict) else None
    early_stop = patience not in (0, None)  # Ultralytics: 0/None = never stop
    if fold == "dev_all" and early_stop:
        errors.append(
            f"dev_all refuses extra.patience: {patience!r} — early stopping would choose the final "
            "epoch on fold0_val, which is inside dev_all_train; use 0"
        )
    for key in ("augment", "extra", "export"):
        need(isinstance(eff.get(key), dict), f"`{key}` must be a mapping")
    if not errors:
        reserved = sorted(RESERVED_EXTRA & (set(eff["extra"]) | set(eff["augment"])))
        need(not reserved, f"{reserved} are set by the runner (top-level keys / --device), "
                           "not under `extra`/`augment`")
        overlap = sorted(set(eff["augment"]) & set(eff["extra"]))
        need(not overlap, f"{overlap} set in both `augment` and `extra`")
        unknown = sorted(set(eff["export"]) - EXPORT_KEYS)
        need(not unknown, f"unknown export key(s) {unknown}; allowed {sorted(EXPORT_KEYS)}")
        opset = eff["export"].get("opset")
        need(isinstance(opset, int) and 11 <= opset <= 20, "export.opset must be an int in 11..20")
        need(eff["export"].get("raw_variant") in RAW_VARIANTS, f"export.raw_variant must be one of {RAW_VARIANTS}")
    parsed = B.parse_run_name(eff["name"])
    if parsed is not None and parsed.fold != fold:
        errors.append(f"name {eff['name']!r} says {parsed.fold} but the config trains fold {fold!r}")
    if errors:
        raise B.ConfigError(f"{eff.get('name')}: " + "; ".join(errors))
    if parsed is None:
        logger.warning(
            f"name {eff['name']!r} does not follow <arch>_<schema>_fold<k> / <arch>_<schema>_devall "
            "(docs/CONTRACTS.md §8.1-8.2): the bundle is its own family and eval cannot pool it "
            "with fold runs — fine for smoke runs only"
        )
    if eff["imgsz"] != TILE_SIZE:
        logger.warning(
            f"imgsz {eff['imgsz']} != tile size {TILE_SIZE}: the export and the inference "
            f"pipeline run on {TILE_SIZE}-px tiles"
        )
    if weights == "best":
        logger.warning(
            "final_weights: best selects the checkpoint on %s_val — the photos this model is "
            "later evaluated on out-of-fold (selection leak); prefer last", fold,
        )
    if early_stop:
        logger.warning(
            "WARNING extra.patience: %r turns early stopping on: the run ends at an epoch chosen on "
            "%s_val — the photos this model is later evaluated on out-of-fold (selection leak; the "
            "out-of-fold evaluation refuses such bundles); keep 0", patience, fold,
        )
    return eff


def runtime_from_env(env=None) -> dict:
    """Non-semantic runtime knobs (workers, cache) from the environment."""
    env = os.environ if env is None else env
    out: dict = {}
    workers = str(env.get(ENV_WORKERS, "")).strip()
    if workers:
        if not workers.isdigit():
            raise B.ConfigError(f"{ENV_WORKERS} must be a non-negative int, got {workers!r}")
        out["workers"] = int(workers)
    cache = str(env.get(ENV_CACHE, "")).strip().lower()
    if cache:
        if cache not in ("disk", "ram", "false", "none"):
            raise B.ConfigError(f"{ENV_CACHE} must be disk|ram|false, got {cache!r}")
        out["cache"] = False if cache in ("false", "none") else cache
    return out


# -------------------------------------------------------------- run state


@dataclass(frozen=True)
class CkptState:
    kind: str                          # missing | corrupt | resumable | finished
    epochs_done: Optional[int] = None  # completed epochs (1-based); None once stripped
    epochs: Optional[int] = None       # the run's target epochs
    stripped: bool = False             # final_eval already stripped the optimizer
    detail: str = ""
    train_args: Optional[dict] = None  # TRAIN_ARG_KEYS the checkpoint trained with (None: unknown)


@dataclass(frozen=True)
class StartPlan:
    kind: str                          # fresh | resume | finished
    source: str                        # none | local | mirror
    epochs_done: Optional[int] = None
    stripped: bool = False
    moved_aside: Optional[Path] = None
    reason: str = ""
    mirror_moved_aside: Optional[Path] = None


# The checkpoint's own arguments a resume trains with (Ultralytics restores
# them from last.pt and ignores ours); they must match the effective config.
TRAIN_ARG_KEYS = ("epochs", "imgsz", "seed", "batch")
OOM_HALVINGS = 8  # Ultralytics halves the batch after a first-epoch OOM (up to 3x per process)
MAX_STRIDE = 32   # YOLO's largest stride: train_args.imgsz is rounded up to a multiple of it


def probe_checkpoint(pt: Path) -> CkptState:
    """Classify an Ultralytics last.pt without building a trainer.

    An OSError while READING the file (Drive mount dropped, EIO) is raised,
    never reported as "corrupt": a checkpoint we could not read is not a bad
    checkpoint. The bytes are parsed from memory, because torch.load itself
    raises OSError(EINVAL) for some truncated files — those are corrupt."""
    pt = Path(pt)
    if not pt.is_file():
        return CkptState("missing")
    import io

    import torch
    from ultralytics.utils.patches import torch_load

    try:
        data = pt.read_bytes()
    except FileNotFoundError:
        return CkptState("missing")
    try:
        ckpt = torch_load(io.BytesIO(data), map_location=torch.device("cpu"))
    except Exception as e:  # truncated write, foreign file, ...
        return CkptState("corrupt", detail=f"{type(e).__name__}: {e}")
    if not isinstance(ckpt, dict) or not (ckpt.get("ema") or ckpt.get("model")):
        return CkptState("corrupt", detail="not an Ultralytics training checkpoint")
    raw_args = ckpt.get("train_args") if isinstance(ckpt.get("train_args"), dict) else {}
    train_args = {k: raw_args[k] for k in TRAIN_ARG_KEYS if k in raw_args}
    epoch = ckpt.get("epoch", -1)
    epochs = raw_args.get("epochs")
    epochs = int(epochs) if isinstance(epochs, (int, float)) else None
    if epoch is None or int(epoch) < 0 or ckpt.get("optimizer") is None:
        return CkptState("finished", None, epochs, stripped=True, train_args=train_args)
    done = int(epoch) + 1
    if epochs is not None and done >= epochs:  # loop done, final_eval never ran
        return CkptState("finished", done, epochs, stripped=False, train_args=train_args)
    return CkptState("resumable", done, epochs, train_args=train_args)


def train_args_mismatch(train_args: Optional[dict], eff: dict) -> Optional[str]:
    """Why a checkpoint's train_args disagree with the effective config, or None.

    batch may be the config's batch halved k times (Ultralytics' first-epoch
    OOM recovery; the bundle records it as batch_effective). An auto batch
    (batch < 1) is resolved by Ultralytics and not compared. imgsz may be
    rounded up to the model stride (check_imgsz stores the rounded value)."""
    if not train_args:
        return None
    for key in ("epochs", "imgsz", "seed"):
        have, want = train_args.get(key), eff.get(key)
        if have is None or want is None or have == want:
            continue
        if key == "imgsz" and isinstance(want, int) and have == -(-want // MAX_STRIDE) * MAX_STRIDE:
            continue
        return f"checkpoint trained with {key}={have!r}, the config says {want!r}"
    have, want = train_args.get("batch"), eff.get("batch")
    if (have is not None and isinstance(want, int) and not isinstance(want, bool) and want >= 1):
        allowed = {max(want >> k, 1) for k in range(OOM_HALVINGS)}
        if have not in allowed:
            return f"checkpoint trained with batch={have!r}, the config says {want!r}"
    return None


def batch_trained(pt: Path) -> Optional[int]:
    """The batch a checkpoint says it trained with (train_args.batch), or None."""
    state = probe_checkpoint(pt)
    batch = (state.train_args or {}).get("batch")
    return int(batch) if isinstance(batch, (int, float)) and not isinstance(batch, bool) else None


def read_sidecar(directory: Path) -> Optional[dict]:
    """zander_run.json, or None when absent / not an attempt log. Any other
    OSError (Drive) is raised: unknown is not the same as missing."""
    try:
        doc = json.loads((Path(directory) / SIDECAR).read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return None
    return doc if isinstance(doc, dict) and isinstance(doc.get("attempts"), list) else None


def write_sidecar(directory: Path, doc: dict) -> None:
    B.atomic_write_text(Path(directory) / SIDECAR, json.dumps(doc, indent=1, ensure_ascii=False))


def _mirror_members(best: bool) -> list[tuple[str, str]]:
    """(run_dir relative path, mirror name) in copy order: results.csv before
    the weights (a mirror's results.csv is never behind its checkpoint), and
    the sidecar LAST — it vouches for the checkpoint next to it (config sha,
    dataset fingerprint), so it may never arrive before that checkpoint."""
    members = [("results.csv", "results.csv"), ("args.yaml", "args.yaml")]
    if best:
        members.append(("weights/best.pt", "best.pt"))
    members.append(("weights/last.pt", "last.pt"))
    members.append((SIDECAR, SIDECAR))
    return members


def mirror_run_dir(run_dir: Path, mirror_run: Path, best: bool = False) -> None:
    for rel, name in _mirror_members(best):
        src = Path(run_dir) / rel
        if src.is_file():
            B.atomic_copy(src, Path(mirror_run) / name)


def restore_from_mirror(mirror_run: Path, run_dir: Path) -> None:
    (Path(run_dir) / "weights").mkdir(parents=True, exist_ok=True)
    for rel, name in _mirror_members(best=True):
        src = Path(mirror_run) / name
        if src.is_file():
            B.atomic_copy(src, Path(run_dir) / rel)


def truncate_results_csv(path: Path, max_epoch: int) -> int:
    """Drop rows past `max_epoch` (1-based). A results.csv can be one epoch ahead
    of last.pt when a process died between the two writes; a resume appends, so
    that row would otherwise appear twice. Returns the number of dropped rows."""
    path = Path(path)
    if not path.is_file():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) < 2:
        return 0
    keep, dropped = [lines[0]], 0
    for line in lines[1:]:
        try:
            epoch = int(float(line.split(",", 1)[0]))
        except ValueError:
            keep.append(line)
            continue
        if epoch <= max_epoch:
            keep.append(line)
        else:
            dropped += 1
    if dropped:
        B.atomic_write_text(path, "".join(keep))
    return dropped


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def move_aside(path: Path, stamp: Optional[str] = None) -> Path:
    """<path> -> <path>.stale-<stamp>[-n]; returns the new location."""
    path = Path(path)
    stamp = stamp or _utc_stamp()
    target = path.with_name(f"{path.name}.stale-{stamp}")
    n = 1
    while target.exists() or target.is_symlink():
        target = path.with_name(f"{path.name}.stale-{stamp}-{n}")
        n += 1
    shutil.move(str(path), str(target))
    return target


def _usable(pt: Path, sidecar_dir: Path, config_sha: str, probe, fingerprint: Optional[str] = None,
            eff: Optional[dict] = None) -> tuple[Optional[CkptState], str]:
    state = probe(pt)
    if state.kind in ("missing", "corrupt"):
        return None, f"checkpoint {state.kind}" + (f" ({state.detail})" if state.detail else "")
    side = read_sidecar(sidecar_dir)
    if side is None:
        return None, f"no {SIDECAR} (provenance unknown)"
    if side.get("config_sha256") != config_sha:
        return None, "config changed (config_sha256 differs)"
    # the config sha does not cover the data: never resume (or bundle) across a tile rebuild
    if fingerprint is not None and side.get("dataset_fingerprint") != fingerprint:
        have = side.get("dataset_fingerprint")
        return None, ("dataset changed (dataset_fingerprint differs)" if have
                      else f"no dataset_fingerprint in {SIDECAR} (data provenance unknown)")
    if eff is not None:
        why = train_args_mismatch(state.train_args, eff)
        if why:
            return None, f"{why} (checkpoint and {SIDECAR} do not belong together)"
    return state, f"{state.kind} checkpoint"


def _has_entries(directory: Path) -> bool:
    directory = Path(directory)
    return directory.is_dir() and any(directory.iterdir())


def _rank(state: CkptState) -> tuple[int, int]:
    return (2, 0) if state.kind == "finished" else (1, state.epochs_done or 0)


def prepare_run_dir(
    run_dir: Path,
    config_sha: str,
    mirror_run: Optional[Path] = None,
    *,
    probe: Optional[Callable[[Path], CkptState]] = None,
    stamp: Optional[str] = None,
    fingerprint: Optional[str] = None,
    eff: Optional[dict] = None,
) -> StartPlan:
    """Decide fresh / resume / finished and make run_dir match the decision.

    Candidates: the local run_dir, then the mirror; a candidate is usable when
    its last.pt is resumable or finished AND its zander_run.json records this
    config sha (and, when given, this dataset `fingerprint`) AND the
    checkpoint's train_args agree with `eff`. The most advanced one wins
    (finished > higher epoch; local on a tie). Using the mirror, or starting
    fresh, first moves an existing run_dir aside; a mirror that is not used and
    not usable is moved aside too (`<mirror>.stale-<stamp>`, never deleted),
    so the new run's mirror never mixes with old files. OSErrors while reading
    a candidate propagate (the attempt fails and is retried)."""
    probe = probe or probe_checkpoint  # looked up at call time
    run_dir = Path(run_dir)
    local, local_why = (None, "no run_dir")
    if run_dir.exists():
        local, local_why = _usable(run_dir / "weights" / "last.pt", run_dir, config_sha, probe,
                                   fingerprint, eff)
    mirror, mirror_why = (None, "no mirror")
    if mirror_run is not None and (Path(mirror_run) / "last.pt").is_file():
        mirror, mirror_why = _usable(Path(mirror_run) / "last.pt", Path(mirror_run), config_sha, probe,
                                     fingerprint, eff)
    reason = f"local: {local_why}; mirror: {mirror_why}"
    moved = mirror_moved = None
    if mirror is None and mirror_run is not None and _has_entries(mirror_run):
        mirror_moved = move_aside(mirror_run, stamp)
    if mirror is not None and (local is None or _rank(mirror) > _rank(local)):
        if run_dir.exists():
            moved = move_aside(run_dir, stamp)
        restore_from_mirror(Path(mirror_run), run_dir)
        state, source = mirror, "mirror"
    elif local is not None:
        state, source = local, "local"
    else:
        if run_dir.exists():
            moved = move_aside(run_dir, stamp)
        return StartPlan("fresh", "none", moved_aside=moved, reason=reason, mirror_moved_aside=mirror_moved)
    if state.kind == "resumable":
        truncate_results_csv(run_dir / "results.csv", state.epochs_done or 0)
        return StartPlan("resume", source, state.epochs_done, False, moved, reason, mirror_moved)
    return StartPlan("finished", source, state.epochs_done, state.stripped, moved, reason, mirror_moved)


def _env_number(name: str, default, cast):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        logger.warning(f"ignoring {name}={raw!r} (not a number); using {default}")
        return default


class EpochHook:
    """`on_model_save` callback: attempt bookkeeping + Drive mirror + test abort."""

    def __init__(self, run_dir: Path, mirror_run: Optional[Path], sidecar: dict,
                 attempt: dict, mirror_best: bool = False):
        self.run_dir = Path(run_dir)
        self.mirror_run = Path(mirror_run) if mirror_run else None
        self.sidecar = sidecar
        self.attempt = attempt
        self.mirror_best = mirror_best
        self.t0 = time.time()
        self.interval = _env_number(ENV_MIRROR_INTERVAL, DEFAULT_MIRROR_INTERVAL_S, float)
        self.abort_after = _env_number(ENV_ABORT, None, int)
        self.last_mirror: Optional[float] = None

    def mirror_now(self, done: int, total: int) -> None:
        if self.mirror_run is None:
            return
        t = time.monotonic()
        try:
            mirror_run_dir(self.run_dir, self.mirror_run, best=self.mirror_best)
            self.last_mirror = time.monotonic()
            logger.info(f"mirror: epoch {done}/{total} -> {self.mirror_run} ({self.last_mirror - t:.1f}s)")
        except Exception as e:  # never let Drive trouble kill a GPU run
            logger.warning(f"mirror copy failed (training continues): {type(e).__name__}: {e}")

    def __call__(self, trainer) -> None:
        done, total = int(trainer.epoch) + 1, int(trainer.epochs)
        self.attempt.update(end_epoch=done, seconds=round(time.time() - self.t0, 1), updated=B.utc_now())
        try:
            write_sidecar(self.run_dir, self.sidecar)
        except Exception as e:
            logger.warning(f"could not update {SIDECAR}: {e}")
        final = done >= total
        abort = self.abort_after is not None and done >= self.abort_after and not final
        due = self.last_mirror is None or final or abort or (
            time.monotonic() - self.last_mirror >= self.interval
        )
        if due:
            self.mirror_now(done, total)
        if abort:
            logger.warning(
                f"{ENV_ABORT}={self.abort_after}: simulating a disconnect after epoch {done} "
                f"(hard exit {ABORT_EXIT_CODE})"
            )
            for h in logging.getLogger().handlers:
                try:
                    h.flush()
                except Exception:
                    pass
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(ABORT_EXIT_CODE)


# ----------------------------------------------------------------- helpers


def _git_meta() -> dict:
    def _run(*args):
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, cwd=Path(__file__).parent
            ).stdout.strip()
        except OSError:
            return "unknown"

    return {
        "git_sha": _run("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(_run("status", "--porcelain")),
    }


def _write_data_yaml(ds_id: str, fold: str, out: Path) -> Path:
    """Runtime data.yaml with absolute paths — survives Colab relocation."""
    yolo_root = (settings.yolo_dir / ds_id).resolve()
    if not yolo_root.exists():
        raise FileNotFoundError(
            f"{yolo_root} missing — run `python cli.py tile --schema <schema>` first "
            "(or set ML_DATA_ROOT to the unzipped dataset location)"
        )
    classes = load_provenance(ds_id)["classes"]
    train = "images/dev_all_train" if fold == "dev_all" else f"images/{fold}_train"
    val = "images/fold0_val" if fold == "dev_all" else f"images/{fold}_val"
    for sub in (train, val):
        if not (yolo_root / sub).is_dir():
            raise FileNotFoundError(
                f"{yolo_root / sub} missing — rebuild the YOLO view with "
                "`python cli.py tile --schema <schema> --no-images`"
            )
    lines = [
        f"path: {json.dumps(str(yolo_root))}",
        f"train: {train}",
        f"val: {val}",
        "names:",
    ] + [f"  {i}: {json.dumps(c, ensure_ascii=False)}" for i, c in enumerate(classes)]
    B.atomic_write_text(out, "\n".join(lines) + "\n")
    return out


def _maybe_float(v: str):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _metrics_from_results_csv(run_dir: Path) -> dict:
    """Per-epoch rows of results.csv, de-duplicated by epoch (last row wins)."""
    results = Path(run_dir) / "results.csv"
    if not results.exists():
        return {"epochs": [], "final": {}}
    by_epoch: dict = {}
    extra_rows = []
    with open(results, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rec = {k.strip(): _maybe_float(v) for k, v in row.items() if k}
            ep = rec.get("epoch")
            if isinstance(ep, float):
                by_epoch[int(ep)] = rec
            else:
                extra_rows.append(rec)
    epochs = [by_epoch[k] for k in sorted(by_epoch)] + extra_rows
    return {"epochs": epochs, "final": epochs[-1] if epochs else {}}


def _epochs_completed(metrics: dict, sidecar: dict) -> Optional[int]:
    eps = [int(r["epoch"]) for r in metrics["epochs"] if isinstance(r.get("epoch"), float)]
    if eps:
        return max(eps)
    ends = [a.get("end_epoch") for a in sidecar.get("attempts", []) if a.get("end_epoch") is not None]
    return max(ends) if ends else None


def _import_yolo():
    """Offline hygiene before the first ultralytics import: no pip auto-installs,
    no telemetry, no experiment-tracker integrations (the registry is the only
    MLflow writer), weights cached in ml_data/pretrained."""
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    settings.pretrained_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    from ultralytics import settings as yolo_settings

    wanted = {k: False for k in ("sync", "mlflow", "clearml", "comet", "dvc", "neptune",
                                 "raytune", "tensorboard", "wandb")}
    wanted["weights_dir"] = str(settings.pretrained_dir)
    for key, value in wanted.items():
        try:  # settings schema varies across versions — non-fatal
            if yolo_settings.get(key) != value:
                yolo_settings.update({key: value})
        except Exception:
            pass
    return YOLO


def _weights_source(model: str) -> str:
    """`yolo26n.pt` -> ml_data/pretrained/yolo26n.pt (downloads land there too);
    a `*.yaml` architecture or an absolute path is used as given."""
    p = Path(model)
    if p.suffix == ".pt" and not p.is_absolute():
        return str(settings.pretrained_dir / p)
    return model


def _gpu_name(device: Optional[str]) -> str:
    import torch

    d = "" if device is None else str(device).strip().lower()
    if d == "cpu":
        return "cpu"
    if d == "mps":
        return "mps"
    if torch.cuda.is_available():
        first = d.split(",")[0].replace("cuda:", "")
        return torch.cuda.get_device_name(int(first) if first.isdigit() else 0)
    return "cpu"


def _versions() -> dict:
    import numpy
    import onnx
    import torch
    import ultralytics

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "onnx": onnx.__version__,
        "numpy": numpy.__version__,
    }
    try:
        import onnxslim

        versions["onnxslim"] = onnxslim.__version__
    except Exception:
        versions["onnxslim"] = None
    return versions


def _pip_freeze() -> str:
    try:
        return subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=300
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"# pip freeze failed: {e}\n"


def _classify_layout(shape: list, nc: int) -> str:
    if len(shape) == 3 and all(isinstance(x, int) for x in shape):
        if shape[2] == 6 and shape[1] != 4 + nc:
            return "e2e"  # (1, max_det, [x1, y1, x2, y2, score, class])
        if shape[1] == 4 + nc:
            return "raw"  # (1, 4 + nc, anchors)
    raise RuntimeError(f"unexpected ONNX output shape {shape} for a {nc}-class detector")


def onnx_info(path: Path, class_names: list) -> dict:
    """Static facts about an exported ONNX (onnx only — no onnxruntime needed on
    Colab): opset, layout, shapes, sha, and the class names Ultralytics baked
    into its metadata, which must equal the dataset's classes."""
    import onnx

    path = Path(path)
    model = onnx.load(str(path))
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), None)

    def shape(vi):
        return [
            d.dim_value if d.HasField("dim_value") else (d.dim_param or None)
            for d in vi.type.tensor_type.shape.dim
        ]

    inits = {i.name for i in model.graph.initializer}
    inputs = [shape(i) for i in model.graph.input if i.name not in inits]
    outputs = [shape(o) for o in model.graph.output]
    props = {p.key: p.value for p in model.metadata_props}
    names = None
    if "names" in props:
        try:
            d = ast.literal_eval(props["names"])
            names = [d[k] for k in sorted(d)]
        except (ValueError, SyntaxError, TypeError):
            names = None
    if names is not None and names != list(class_names):
        raise RuntimeError(
            f"{path.name}: model class names {names} != dataset classes {list(class_names)}"
        )
    return {
        "file": path.name,
        "sha256": B.sha256_file(path),
        "opset": opset,
        "layout": _classify_layout(outputs[0], len(class_names)),
        "output_shapes": outputs,
        "input_shapes": inputs,
        "bytes": path.stat().st_size,
        "class_names": names,
    }


def export_onnx(weights_pt: Path, export_dir: Path, *, opset: int, raw_variant: str,
                class_names: list) -> tuple[dict, Optional[dict]]:
    """model.onnx (default head) + model_raw.onnx for end-to-end models."""
    YOLO = _import_yolo()
    export_dir = Path(export_dir)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    # FP32 (Ultralytics' default; its `half` flag is deprecated in 8.4), static
    # 640x640 input, graph slimmed by onnxslim, traced on CPU for determinism.
    common = dict(format="onnx", nms=False, imgsz=TILE_SIZE, opset=int(opset), dynamic=False,
                  simplify=True, device="cpu")
    produced = YOLO(str(weights_pt)).export(**common)
    main_path = export_dir / "model.onnx"
    shutil.move(str(produced), main_path)
    main = onnx_info(main_path, class_names)
    raw = None
    if raw_variant == "auto" and main["layout"] == "e2e":
        produced = YOLO(str(weights_pt)).export(**common, end2end=False)
        raw_path = export_dir / "model_raw.onnx"
        shutil.move(str(produced), raw_path)
        raw = onnx_info(raw_path, class_names)
        if raw["layout"] != "raw":
            raise RuntimeError(f"end2end=False export produced layout {raw['layout']!r}")
    return main, raw


def _move_stale_bundle(run_root: Path, name: str) -> None:
    zip_path = run_root / f"{name}_bundle.zip"
    for p in (zip_path, B.sha_path_for(zip_path)):
        if p.exists():
            logger.warning(f"moving a stale local bundle aside: {move_aside(p)}")


# --------------------------------------------------------------------- run


def run(
    config_path: Path,
    device: Optional[str] = None,
    overrides: Optional[dict] = None,
    mirror_dir: Optional[Path] = None,
) -> Path:
    """Train (or resume / finish) one config and return its bundle zip."""
    config_path = Path(config_path)
    overrides = dict(overrides or {})
    eff = resolve_config(B.load_config(config_path), overrides)
    cfg_sha = B.config_sha256(eff)
    name, fold, ds_id = eff["name"], eff["fold"], eff["dataset_id"]
    final_weights = eff["final_weights"]
    runtime = runtime_from_env()
    provenance = load_provenance(ds_id)
    parsed = B.parse_run_name(name)
    if parsed is not None and provenance.get("schema") != parsed.schema:
        raise B.ConfigError(
            f"{name}: name says schema {parsed.schema!r} but {ds_id} is {provenance.get('schema')!r}"
        )

    run_root = settings.run_dirs_dir
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / name
    mirror_run = Path(mirror_dir) / name if mirror_dir else None
    data_yaml = _write_data_yaml(ds_id, fold, run_root / f"{name}_data.yaml")

    YOLO = _import_yolo()
    fingerprint = provenance.get("dataset_fingerprint")
    plan = prepare_run_dir(run_dir, cfg_sha, mirror_run, fingerprint=fingerprint, eff=eff)
    logger.info(
        f"{name}: {plan.kind} (source {plan.source}, epochs done {plan.epochs_done}; {plan.reason})"
    )
    if plan.moved_aside:
        logger.warning(f"previous run_dir moved aside -> {plan.moved_aside}")
    if plan.mirror_moved_aside:
        logger.warning(f"unusable mirror moved aside -> {plan.mirror_moved_aside}")

    sidecar = read_sidecar(run_dir) if plan.kind != "fresh" else None
    if sidecar is None:
        sidecar = {"format": 1, "run_name": name, "config_sha256": cfg_sha,
                   "dataset_fingerprint": fingerprint, "created": B.utc_now(), "attempts": []}
    for a in sidecar["attempts"]:
        if a.get("status") == "running":  # its process is gone
            a["status"] = "interrupted"

    if plan.kind in ("fresh", "resume"):
        _move_stale_bundle(run_root, name)
        resumed = plan.kind == "resume"
        start = (plan.epochs_done or 0) if resumed else 0
        attempt = {
            "n": len(sidecar["attempts"]) + 1, "gpu": _gpu_name(device), "device": device,
            "started": B.utc_now(), "seconds": 0.0, "resumed": resumed,
            "start_epoch": start, "end_epoch": start, "workers": runtime.get("workers"),
            "cache": runtime.get("cache"), "host": platform.node(), "status": "running",
        }
        sidecar["attempts"].append(attempt)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_sidecar(run_dir, sidecar)
        hook = EpochHook(run_dir, mirror_run, sidecar, attempt, mirror_best=final_weights == "best")
        if resumed:
            model = YOLO(str(run_dir / "weights" / "last.pt"))
            kwargs = dict(resume=True, data=str(data_yaml), save_dir=str(run_dir))
        else:
            model = YOLO(_weights_source(eff["model"]))
            kwargs = dict(
                data=str(data_yaml), epochs=eff["epochs"], imgsz=eff["imgsz"],
                batch=eff["batch"], seed=eff["seed"], deterministic=eff["deterministic"],
                project=str(run_root), name=name, exist_ok=True,
                **eff["augment"], **eff["extra"],
            )
        kwargs.update(runtime)
        if device is not None:
            kwargs["device"] = device
        model.add_callback("on_model_save", hook)
        logger.info(
            f"training {name}: model={eff['model']} fold={fold} epochs={eff['epochs']} "
            f"{'resume from epoch ' + str(start) if resumed else 'fresh'} runtime={runtime}"
        )
        hook.t0 = time.time()
        model.train(**kwargs)
        metrics_now = _metrics_from_results_csv(run_dir)
        attempt.update(
            seconds=round(time.time() - hook.t0, 1), status="finished", updated=B.utc_now(),
            end_epoch=_epochs_completed(metrics_now, sidecar),
        )
        write_sidecar(run_dir, sidecar)
        if mirror_run is not None:  # the mirror's attempt log should say "finished" too
            try:
                B.atomic_copy(run_dir / SIDECAR, mirror_run / SIDECAR)
            except Exception as e:
                logger.warning(f"mirror sidecar update failed: {e}")
    else:
        logger.info(f"{name}: training already finished — exporting and bundling only")
        write_sidecar(run_dir, sidecar)
        if not plan.stripped:  # loop finished but final_eval never stripped it
            from ultralytics.utils.torch_utils import strip_optimizer

            for w in ("last.pt", "best.pt"):
                if (run_dir / "weights" / w).is_file():
                    strip_optimizer(run_dir / "weights" / w)

    weights_pt = run_dir / "weights" / f"{final_weights}.pt"
    if not weights_pt.is_file():
        raise FileNotFoundError(f"{weights_pt} missing — cannot bundle final_weights={final_weights}")
    if sidecar.get("dataset_fingerprint") != fingerprint:  # meta below is stamped with today's data
        raise RuntimeError(
            f"{name}: the checkpoint was trained on dataset_fingerprint "
            f"{sidecar.get('dataset_fingerprint')!r}, today's tiles are {fingerprint!r} — not bundled"
        )
    batch_effective = batch_trained(run_dir / "weights" / "last.pt")
    if batch_effective is not None and batch_effective != eff["batch"]:
        logger.warning(
            f"WARNING {name}: trained with batch {batch_effective}, but the config says {eff['batch']} "
            "(Ultralytics halves the batch after a CUDA out-of-memory error). meta.json records "
            "batch_effective; this model is NOT config-equivalent to its fold siblings"
        )

    t_export = time.time()
    onnx_main, onnx_raw = export_onnx(
        weights_pt, run_dir / "export", opset=eff["export"]["opset"],
        raw_variant=eff["export"]["raw_variant"], class_names=provenance["classes"],
    )
    export_seconds = round(time.time() - t_export, 1)
    metrics = _metrics_from_results_csv(run_dir)
    meta = B.build_meta(
        config=eff, provenance=provenance, framework=FRAMEWORK, adapter=FRAMEWORK,
        weights=final_weights, epochs_completed=_epochs_completed(metrics, sidecar),
        attempts=sidecar["attempts"], onnx=onnx_main, onnx_raw=onnx_raw,
        versions=_versions(), overrides=B.to_jsonable(overrides), git=_git_meta(),
        extra={
            "deterministic": eff["deterministic"],
            "final_weights": final_weights,
            "batch_effective": batch_effective,
            "runtime": B.to_jsonable(runtime),
            "export_seconds": export_seconds,
            "toolchain": f"{settings.toolchain_name} {settings.toolchain_version}",
        },
    )
    files = {
        "config.yaml": B.dump_config(eff),
        "meta.json": json.dumps(meta, indent=1, ensure_ascii=False),
        "metrics.json": json.dumps(metrics, indent=1),
        "pip-freeze.txt": _pip_freeze(),
        "model.onnx": run_dir / "export" / "model.onnx",
        f"{final_weights}.pt": weights_pt,
    }
    if onnx_raw is not None:
        files["model_raw.onnx"] = run_dir / "export" / "model_raw.onnx"
    if (run_dir / "args.yaml").is_file():
        files["args.yaml"] = run_dir / "args.yaml"
    zip_path, sha = B.write_bundle(run_root / f"{name}_bundle.zip", files)
    logger.info(
        f"bundle ready: {zip_path} (sha256 {sha[:12]}, {meta['epochs_completed']} epochs, "
        f"{meta['train_seconds'] / 60:.1f} train-min over {len(meta['attempts'])} attempt(s)) "
        "— import with `python cli.py import-run`"
    )
    return zip_path
