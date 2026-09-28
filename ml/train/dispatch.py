"""
Training entry point: one run config -> one bundle zip, whatever the framework
(docs/CONTRACTS.md §8.1). Run from ml/:

    python train/dispatch.py --config configs/matrix/yolo26n_def3_fold0.yaml --device 0
    python train/dispatch.py --config configs/smoke_def3.yaml --device cpu
    python train/dispatch.py --config ... --set epochs=60 --set extra.val=false \
        --mirror-dir /content/drive/MyDrive/ZanderML/mirror --workers 8 --cache disk

The config's `framework` picks the runner: `ultralytics` (default) ->
train/ultralytics_run.run; `deim` -> train/deim_run.run_from_config (WP-deim).
`--set KEY=VALUE` overrides use dotted keys (YAML-parsed values) and end up in
the bundle's effective config.yaml; identity keys (name, model, fold, ...)
cannot be overridden. `--workers/--cache` are runtime knobs (recorded per
attempt, never part of the config). A mirror dir enables resume after a
disconnect (see train/ultralytics_run.py).

Exit codes: 0 = bundle written (the last stdout line is `ZANDER_BUNDLE=<zip>`),
1 = training/runtime failure, 2 = configuration or dataset error (retrying
won't help), 130 = interrupted. train/matrix.py relies on these.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

ML_DIR = Path(__file__).resolve().parents[1]
_TRAIN_DIR = Path(__file__).resolve().parent
# Script mode puts ml/train on sys.path[0]; import everything as `train.*` from ml/.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _TRAIN_DIR]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from train.bundle import ConfigError, load_config, parse_overrides  # noqa: E402

logger = logging.getLogger("ml.dispatch")

FRAMEWORKS = ("ultralytics", "deim")
DEFAULT_FRAMEWORK = "ultralytics"
BUNDLE_MARKER = "ZANDER_BUNDLE="
EXIT_OK, EXIT_FAILED, EXIT_CONFIG, EXIT_INTERRUPTED = 0, 1, 2, 130


def framework_of(cfg: dict) -> str:
    fw = cfg.get("framework") or DEFAULT_FRAMEWORK
    if fw not in FRAMEWORKS:
        raise ConfigError(f"unknown framework {fw!r}; one of {FRAMEWORKS}")
    return fw


def _deim_module():
    try:
        from train import deim_run
    except ImportError as e:  # pragma: no cover - depends on WP-deim
        raise NotImplementedError(f"framework 'deim': train/deim_run.py cannot be imported: {e}") from e
    if not hasattr(deim_run, "run_from_config"):
        raise NotImplementedError(
            "framework 'deim' is not wired yet: train/deim_run.py has no "
            "run_from_config(config_path, device=None, overrides=None, mirror_dir=None) "
            "(docs/CONTRACTS.md §8.1; WP-deim adds it)"
        )
    return deim_run


def resolve(config_path: Path, overrides: Optional[dict] = None) -> Optional[dict]:
    """The effective config a run would train with — without training.

    Raises ConfigError for an invalid config and NotImplementedError when the
    framework's runner does not exist yet. Returns None when the runner exists
    but offers no resolver (a DEIM runner may add
    `resolve_config(raw, overrides) -> dict`); callers then skip sha checks."""
    raw = load_config(config_path)
    fw = framework_of(raw)
    if fw == "ultralytics":
        from train import ultralytics_run

        return ultralytics_run.resolve_config(raw, overrides)
    resolver = getattr(_deim_module(), "resolve_config", None)
    return resolver(raw, overrides) if resolver else None


def run(
    config_path: Path,
    device: Optional[str] = None,
    overrides: Optional[dict] = None,
    mirror_dir: Optional[Path] = None,
) -> Path:
    """Train one config with its framework's runner; returns the bundle zip."""
    config_path = Path(config_path)
    fw = framework_of(load_config(config_path))
    mirror_dir = Path(mirror_dir) if mirror_dir else None
    if fw == "ultralytics":
        from train import ultralytics_run

        return Path(ultralytics_run.run(config_path, device=device, overrides=overrides,
                                        mirror_dir=mirror_dir))
    deim_run = _deim_module()
    return Path(deim_run.run_from_config(config_path, device=device, overrides=overrides,
                                         mirror_dir=mirror_dir))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train/dispatch.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="run config YAML (e.g. configs/matrix/*.yaml)")
    p.add_argument("--device", default=None, help="e.g. 0, cpu (default: framework auto)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted config override, repeatable (e.g. epochs=60, extra.val=false)")
    p.add_argument("--mirror-dir", default=None,
                   help="resume mirror root (e.g. MyDrive/ZanderML/mirror); <dir>/<name>/ per run")
    p.add_argument("--workers", type=int, default=None, help="runtime: dataloader workers")
    p.add_argument("--cache", default=None, choices=["disk", "ram", "false"],
                   help="runtime: image cache (disk keeps runs deterministic)")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from config.settings import settings

        level = settings.log_level
    except Exception:
        level = "INFO"
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if args.workers is not None:
        os.environ["ZANDER_WORKERS"] = str(args.workers)
    if args.cache is not None:
        os.environ["ZANDER_CACHE"] = args.cache
    config = Path(args.config)
    if not config.is_file() and (ML_DIR / config).is_file():
        config = ML_DIR / config
    try:
        overrides = parse_overrides(args.overrides)
        zip_path = run(config, device=args.device, overrides=overrides,
                       mirror_dir=Path(args.mirror_dir) if args.mirror_dir else None)
    except KeyboardInterrupt:
        logger.error("interrupted")
        return EXIT_INTERRUPTED
    except (ConfigError, FileNotFoundError, NotImplementedError) as e:
        logger.error(f"configuration/data error (not retryable): {e}")
        return EXIT_CONFIG
    except Exception:
        logger.error("training failed:\n" + traceback.format_exc())
        return EXIT_FAILED
    print(f"{BUNDLE_MARKER}{zip_path}", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
