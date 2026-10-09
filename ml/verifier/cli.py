r"""
Verifier CLI — run from ml/ (like every toolchain step):

    python verifier/cli.py mine --family yolo26n_def3 --run-name v1
    python verifier/cli.py crops --run-name v1
    python verifier/cli.py crossfit --run-name v1 --device mps [--operating-point op.json]
    python verifier/cli.py export --run-name v1
    python verifier/cli.py exit --run-name v1
    python verifier/cli.py all --family yolo26n_def3 --run-name v1 --device mps
    python verifier/cli.py cascade-eval --run-name v1 --model m.onnx --bucket test \
        --operating-point op.json

`python cli.py verifier <args>` (ml/cli.py) delegates here: main(argv) -> int.
Smoke (no GPU, the smoke fold runs share one model, so they are NOT out-of-fold):

    PYTORCH_ENABLE_MPS_FALLBACK=1 python verifier/cli.py all \
        --runs smoke_def3_fold0 smoke_def3_fold1 smoke_def3_fold2 --allow-non-oof \
        --run-name smoke_v --device mps --fraction 0.2 --epochs 1 \
        --operating-point ../ml_data/eval/operating_point_def3_smoke_def3_fold0.json
"""
from __future__ import annotations

import sys
from pathlib import Path

# `python verifier/cli.py` puts ml/verifier/ first on sys.path, where
# verifier/config.py would shadow the ml/config package: put ml/ first instead.
_HERE = Path(__file__).resolve().parent
_ML = _HERE.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

import argparse  # noqa: E402
import logging  # noqa: E402

logger = logging.getLogger("ml.verifier.cli")


def _overrides(args) -> dict:
    import yaml

    out = {}
    for item in getattr(args, "set", None) or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        out[key.strip()] = yaml.safe_load(value.strip())  # numbers, lists, null, true/false
    if getattr(args, "epochs", None) is not None:
        out["train.epochs"] = args.epochs
    return out


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-name", default="v1", help="verifier run name (ml_data/verifier/<name>/)")
    p.add_argument("--config", default=None, help="verifier config YAML (default: the pre-registered one)")
    p.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE",
                   help="config override (recorded; makes the run exploratory — also at exit "
                        "time: the exit decision is then invalid-smoke)")


def _add_mine(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--runs", nargs=3, metavar="EVAL_RUN", help="the three own-fold eval runs")
    g.add_argument("--family", help="detector family: resolves to <F>_fold0..2")
    p.add_argument("--allow-non-oof", action="store_true",
                   help="SMOKE ONLY: accept runs that are not out-of-fold (stamps the run smoke)")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="deterministic photo fraction per fold (smoke speed-up; < 1 is never shippable)")


def _add_crossfit(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="auto", help="auto | mps | cpu | cuda")
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs (exploratory)")
    p.add_argument("--operating-point", default=None,
                   help="the family's operating_point.json (default: ml_data/eval/operating_point_<schema>_<family>.json)")
    p.add_argument("--no-eval-runs", action="store_true", help="skip writing the cascade eval runs")


def _add_exit(p: argparse.ArgumentParser) -> None:
    p.add_argument("--latency-photos", type=int, default=None,
                   help="latency bench photos (default: the config's 20; another count makes "
                        "the exit exploratory)")
    p.add_argument("--threads", type=int, default=0,
                   help="ORT intra-op threads (0 = ORT default, pre-registered; anything else "
                        "makes the exit exploratory)")


def build_parser() -> argparse.ArgumentParser:
    from verifier.common import CASCADE_BUCKETS

    parser = argparse.ArgumentParser(prog="verifier", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mine", help="mine out-of-fold candidates from the fold eval runs")
    _add_common(p)
    _add_mine(p)
    p = sub.add_parser("crops", help="cut the 160-px windows cache")
    _add_common(p)
    p = sub.add_parser("crossfit", help="nested cross-fit: verifiers, stackers, thresholds, cascade runs")
    _add_common(p)
    _add_crossfit(p)
    p = sub.add_parser("export", help="verifier.onnx (parity-checked) + stacker.json")
    _add_common(p)
    p = sub.add_parser("exit", help="pre-registered exit rule -> exit_report.{md,json}")
    _add_common(p)
    _add_exit(p)
    p = sub.add_parser("all", help="mine -> crops -> crossfit -> export -> exit")
    _add_common(p)
    _add_mine(p)
    _add_crossfit(p)
    _add_exit(p)
    p = sub.add_parser("cascade-eval", help="detector + verifier on test / probe_xprod / future")
    _add_common(p)
    p.add_argument("--model", required=True, help="detector .onnx")
    p.add_argument("--adapter", default="ultralytics", choices=["ultralytics", "deim"])
    p.add_argument("--schema", default="def3", choices=["def3", "nal1"])
    p.add_argument("--bucket", required=True, choices=list(CASCADE_BUCKETS),
                   help="never a dev fold: stage 2 was fitted on them (the cross-fit runs "
                        "are the out-of-fold numbers)")
    p.add_argument("--operating-point", default=None)
    p.add_argument("--eval-run-name", default=None)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--second-look", default=None,
                   help="test bucket only: the recorded reason for a second look")
    return parser


def main(argv: list[str] | None = None) -> int:
    from config.settings import settings

    if not logging.getLogger().handlers:
        logging.basicConfig(level=settings.log_level,
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)
    from verifier.common import Refusal
    from verifier.config import load_config

    cfg = load_config(Path(args.config) if args.config else None, _overrides(args))
    op = Path(args.operating_point) if getattr(args, "operating_point", None) else None
    try:
        if args.cmd in ("mine", "all"):
            from verifier import mine

            mine.mine(runs=args.runs, family=args.family, allow_non_oof=args.allow_non_oof,
                      vrun=args.run_name, cfg=cfg, fraction=args.fraction)
        if args.cmd in ("crops", "all"):
            from verifier import crops

            crops.build(args.run_name)
        if args.cmd in ("crossfit", "all"):
            from verifier import crossfit

            crossfit.run(args.run_name, device=args.device, cfg=cfg, operating_point=op,
                         write_eval_runs=not args.no_eval_runs)
        if args.cmd in ("export", "all"):
            from verifier import export_onnx

            export_onnx.export(args.run_name)
        if args.cmd in ("exit", "all"):
            from verifier import exit_rule

            exit_rule.run(args.run_name, cfg=cfg, threads=args.threads,
                          latency_photos=args.latency_photos)
        if args.cmd == "cascade-eval":
            from verifier import cascade_eval

            cascade_eval.run(args.run_name, Path(args.model), adapter=args.adapter,
                             schema=args.schema, bucket=args.bucket, operating_point=op,
                             run_name=args.eval_run_name, threads=args.threads,
                             second_look=args.second_look)
    except Refusal as e:
        logger.error(f"refused: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
