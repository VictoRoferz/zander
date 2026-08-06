"""
zander-ml CLI — run every toolchain step from inside ml/:

    cd ml
    python cli.py ingest            # manifest from photos + label snapshot (+gates)
    python cli.py split             # groups + frozen grouped split (test-v1 + folds)
    python cli.py tile              # 640-px tiles + COCO GT + YOLO view (+--zip)
    python cli.py report            # committed dataset card
    python cli.py latency           # tiled-inference latency bench for an ONNX model
    python cli.py eval              # stitched full-photo evaluation of an ONNX model
    python cli.py import-run        # run_dir bundle -> local MLflow + results.csv
    python cli.py train             # Ultralytics training (local smoke or Colab)

Steps are idempotent; `split` refuses to overwrite the frozen split file.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import settings  # noqa: E402

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zander-ml", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="build the committed manifest")
    p.add_argument("--no-verify-sha", action="store_true")
    p.add_argument("--no-gates", action="store_true")

    p = sub.add_parser("split", help="build groups + the frozen grouped split")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--regroup", action="store_true", help="recompute groups.json")

    p = sub.add_parser("tile", help="tiles + COCO GT + YOLO view")
    p.add_argument("--no-images", action="store_true", help="only rebuild GT files")
    p.add_argument("--zip", action="store_true", help="also build the Colab zip")

    sub.add_parser("report", help="write the dataset card")

    p = sub.add_parser("latency", help="tiled-inference latency bench")
    p.add_argument("--model", required=True, help="path to .onnx")
    p.add_argument("--adapter", default="ultralytics", choices=["ultralytics", "deim"])
    p.add_argument("--photos", type=int, default=10)
    p.add_argument("--threads", type=int, default=0, help="0 = onnxruntime default")

    p = sub.add_parser("eval", help="stitched full-photo evaluation")
    p.add_argument("--model", required=True, help="path to .onnx")
    p.add_argument("--adapter", default="ultralytics", choices=["ultralytics", "deim"])
    p.add_argument("--bucket", default="fold0", help="fold0|fold1|fold2|dev|test")
    p.add_argument("--run-name", default=None)
    p.add_argument("--conf", type=float, default=0.001)

    p = sub.add_parser("import-run", help="import a run_dir bundle into MLflow")
    p.add_argument("bundle", help="run_dir directory or .zip")

    p = sub.add_parser("train", help="Ultralytics training from a run config YAML")
    p.add_argument("--config", required=True, help="ml/configs/*.yaml")
    p.add_argument("--device", default=None, help="e.g. cpu, 0")

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        from dataprep import ingest

        ingest.run(
            verify_sha=not args.no_verify_sha, enforce_gates=not args.no_gates
        )
    elif args.cmd == "split":
        from dataprep import grouping, split

        groups_path = settings.datasets_dir / "groups.json"
        if args.regroup or not groups_path.exists():
            grouping.run()
        split.run(k=args.k)
    elif args.cmd == "tile":
        from dataprep import tiler, yolo_convert

        tiler.run(write_images=not args.no_images)
        yolo_convert.run()
        if args.zip:
            yolo_convert.export_zip()
    elif args.cmd == "report":
        from dataprep import report

        report.run()
    elif args.cmd == "latency":
        from eval import latency

        latency.run(
            model_path=Path(args.model),
            adapter_name=args.adapter,
            n_photos=args.photos,
            threads=args.threads,
        )
    elif args.cmd == "eval":
        from eval import run_eval

        run_eval.run(
            model_path=Path(args.model),
            adapter_name=args.adapter,
            bucket=args.bucket,
            run_name=args.run_name,
            conf_floor=args.conf,
        )
    elif args.cmd == "import-run":
        import registry

        registry.import_run(Path(args.bundle))
    elif args.cmd == "train":
        from train import ultralytics_run

        ultralytics_run.run(config_path=Path(args.config), device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
