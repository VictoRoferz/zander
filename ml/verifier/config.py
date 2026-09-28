"""
Verifier configuration: the committed, pre-registered YAML
(verifier/configs/mnv3s_2crop.yaml) plus explicit dotted overrides
({"train.epochs": 1}). Every override is recorded; a run with overrides or a
non-default config file is exploratory and can never reach a "ship" decision
(see exit_rule.py).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from infer.verifier import JITTER
from verifier.common import sha256_file

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "mnv3s_2crop.yaml"


@dataclass(frozen=True)
class ModelCfg:
    backbone: str = "mobilenet_v3_small"
    pretrained: str = "imagenet1k_v1"
    embed_dim: int = 128
    dropout: float = 0.2


@dataclass(frozen=True)
class CandidatesCfg:
    top_k: int = 50
    candidate_floor: float = 0.001
    iou_positive: float = 0.3
    ignore_band: tuple = (0.1, 0.3)
    min_pos_per_class: int = 30
    smoke_inject_gt: bool = True


@dataclass(frozen=True)
class AugmentCfg:
    jitter: int = 8
    d4: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.3
    hue: float = 0.02


@dataclass(frozen=True)
class TrainCfg:
    epochs: int = 15
    batch_size: int = 256
    lr_head: float = 1e-3
    lr_backbone: float = 3e-4
    weight_decay: float = 1e-4
    schedule: str = "cosine"
    positive_share: float = 0.25
    freeze_bn: bool = True
    seed: int = 20260928


@dataclass(frozen=True)
class StackerCfg:
    C: float = 1.0
    hi_passthrough: Union[str, float, None] = "auto"
    hi_min_pos: int = 5


@dataclass(frozen=True)
class ExitCfg:
    target_sensitivity: float = 0.95
    fa_reduction: float = 0.30
    latency_verifier_p50_ms: float = 200.0
    latency_total_p50_s: float = 1.0
    latency_photos: int = 20
    latency_warmup: int = 2
    bootstrap_n: int = 1000
    bootstrap_seed: int = 0


SECTIONS = {
    "model": ModelCfg, "candidates": CandidatesCfg, "augment": AugmentCfg,
    "train": TrainCfg, "stacker": StackerCfg, "exit": ExitCfg,
}


@dataclass(frozen=True)
class VerifierConfig:
    name: str
    model: ModelCfg
    candidates: CandidatesCfg
    augment: AugmentCfg
    train: TrainCfg
    stacker: StackerCfg
    exit: ExitCfg
    path: str = ""
    sha256: str = ""
    overrides: dict = field(default_factory=dict)

    @property
    def is_default(self) -> bool:
        """The committed pre-registered config, unmodified."""
        return (
            not self.overrides
            and Path(self.path).resolve() == DEFAULT_CONFIG.resolve()
        )

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["candidates"]["ignore_band"] = list(self.candidates.ignore_band)
        return d


def _coerce(cls, section: str, raw: dict) -> Any:
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"unknown keys in config section {section!r}: {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        default = known[name].default
        if value is None and name != "hi_passthrough":
            raise ValueError(f"config {section}.{name} cannot be null")
        if name == "ignore_band":
            value = tuple(float(v) for v in value)
        elif name == "hi_passthrough":
            if value is None or (isinstance(value, str) and value.lower() in ("null", "none")):
                value = None
            elif value != "auto":
                value = float(value)
        elif isinstance(default, bool):
            if isinstance(value, str):
                value = value.lower() in ("1", "true", "yes", "on")
            value = bool(value)
        elif isinstance(default, int):
            value = int(value)
        elif isinstance(default, float):
            value = float(value)
        kwargs[name] = value
    return cls(**kwargs)


def load_config(path: Optional[Path] = None, overrides: Optional[dict] = None) -> VerifierConfig:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    overrides = dict(overrides or {})
    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        if section not in SECTIONS or not key:
            raise ValueError(f"override {dotted!r} must be <section>.<key>, sections {sorted(SECTIONS)}")
        raw.setdefault(section, {})[key] = value
    unknown = set(raw) - set(SECTIONS) - {"name"}
    if unknown:
        raise ValueError(f"unknown config sections {sorted(unknown)}")
    cfg = VerifierConfig(
        name=str(raw.get("name", path.stem)),
        path=str(path),
        sha256=sha256_file(path),
        overrides=dict(overrides),
        **{s: _coerce(cls, s, raw.get(s) or {}) for s, cls in SECTIONS.items()},
    )
    _validate(cfg)
    return cfg


def _validate(cfg: VerifierConfig) -> None:
    errs = []
    c, a, t = cfg.candidates, cfg.augment, cfg.train
    if cfg.model.backbone != "mobilenet_v3_small":
        errs.append("model.backbone: only mobilenet_v3_small is implemented")
    if cfg.model.pretrained not in ("imagenet1k_v1", "none"):
        errs.append("model.pretrained must be imagenet1k_v1 or none")
    if c.top_k < 1:
        errs.append("candidates.top_k must be >= 1")
    if not (0 < c.candidate_floor < 1):
        errs.append("candidates.candidate_floor must be in (0, 1)")
    lo, hi = c.ignore_band
    if not (0 <= lo <= hi <= c.iou_positive <= 1):
        errs.append("need 0 <= ignore_band[0] <= ignore_band[1] <= iou_positive <= 1")
    if not (0 <= a.jitter <= JITTER):
        errs.append(f"augment.jitter must be in [0, {JITTER}] (the crop geometry's margin)")
    if t.epochs < 1 or t.batch_size < 1:
        errs.append("train.epochs and train.batch_size must be >= 1")
    if not (0 < t.positive_share < 1):
        errs.append("train.positive_share must be in (0, 1)")
    if t.schedule not in ("cosine", "constant"):
        errs.append("train.schedule must be cosine or constant")
    h = cfg.stacker.hi_passthrough
    if not (h is None or h == "auto" or (isinstance(h, float) and 0 < h <= 1)):
        errs.append("stacker.hi_passthrough must be auto, null or a score in (0, 1]")
    if cfg.stacker.C <= 0:
        errs.append("stacker.C must be > 0")
    if errs:
        raise ValueError("invalid verifier config: " + "; ".join(errs))
