"""
zander-ml CLI — run every toolchain step from inside ml/:

    cd ml
    python cli.py attributes        # committed photo-attributes table from the audit
    python cli.py ingest            # manifest from photos + label snapshot + attributes (+gates)
    python cli.py split             # board-aware groups + frozen split_v2
    python cli.py tile --schema def3  # 640-px tiles + offline augmentation + COCO + YOLO view (+--zip)
    python cli.py qa --schema def3  # gallery of augmented training tiles (eyeball once per build)
    python cli.py report            # committed dataset card
    python cli.py latency           # tiled-inference latency bench for an ONNX model
    python cli.py eval              # class-aware photo-level evaluation of an ONNX model
    python cli.py choose-op         # freeze the operating threshold from out-of-fold eval runs
    python cli.py compare           # paired comparison of two eval runs on the same bucket
    python cli.py import-run        # run_dir bundle -> local MLflow
    python cli.py train             # Ultralytics training (local smoke or Colab)

Steps are idempotent; `split` refuses to overwrite a frozen split file.
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

SCHEMA_CHOICES = ["def3", "nal1"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zander-ml", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("attributes", help="build ml/datasets/photo_attributes_v1.csv")

    p = sub.add_parser("ingest", help="build the committed manifest")
    p.add_argument("--no-verify-sha", action="store_true")
    p.add_argument("--no-gates", action="store_true")

    p = sub.add_parser("split", help="build board-aware groups + the frozen split")
    p.add_argument("--k", type=int, default=3)

    p = sub.add_parser("tile", help="tiles + offline augmentation + COCO GT + YOLO view")
    p.add_argument("--schema", default="def3", choices=SCHEMA_CHOICES)
    p.add_argument("--no-images", action="store_true", help="only rebuild GT files")
    p.add_argument("--zip", action="store_true", help="also build the Colab zip")
    # ablation switches (non-default values get a dataset_id suffix)
    p.add_argument("--no-d4", action="store_true", help="ablation: no D4 orientation")
    p.add_argument("--ct-p", type=float, default=None, help="ablation: colour-transfer prob.")
    p.add_argument("--no-handheld-cap", action="store_true", help="ablation: keep all handheld negatives")
    p.add_argument("--fixed-k", type=int, default=None, help="ablation: v1 fixed K instead of RFS")

    p = sub.add_parser("qa", help="gallery of augmented training tiles")
    p.add_argument("--schema", default="def3", choices=SCHEMA_CHOICES)
    p.add_argument("--n", type=int, default=48)

    p = sub.add_parser("report", help="write the dataset card")
    p.add_argument("--schema", default="def3", choices=SCHEMA_CHOICES)

    p = sub.add_parser("latency", help="tiled-inference latency bench")
    p.add_argument("--model", required=True, help="path to .onnx")
    p.add_argument("--adapter", default="ultralytics", choices=["ultralytics", "deim"])
    p.add_argument("--photos", type=int, default=10)
    p.add_argument("--threads", type=int, default=0, help="0 = onnxruntime default")

    p = sub.add_parser("eval", help="class-aware photo-level evaluation")
    p.add_argument("--model", required=True, help="path to .onnx")
    p.add_argument("--adapter", default="ultralytics", choices=["ultralytics", "deim"])
    p.add_argument("--schema", default="def3", choices=SCHEMA_CHOICES)
    p.add_argument(
        "--bucket", default="fold0",
        help="fold0|fold1|fold2|dev|b_holdout|test|probe_xprod|future",
    )
    p.add_argument("--operating-point", default=None,
                   help="operating_point.json (REQUIRED for test/probe_xprod/future)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--conf", type=float, default=0.001)

    p = sub.add_parser("choose-op", help="freeze t* from out-of-fold eval runs")
    p.add_argument("runs", nargs="+", help="eval run names (one per fold)")
    p.add_argument("--target-sensitivity", type=float, default=0.95)
    p.add_argument("--out", default=None)

    p = sub.add_parser("compare", help="paired comparison of two eval runs")
    p.add_argument("run_a")
    p.add_argument("run_b")

    p = sub.add_parser("import-run", help="import a run_dir bundle into MLflow")
    p.add_argument("bundle", help="run_dir directory or .zip")

    p = sub.add_parser("train", help="Ultralytics training from a run config YAML")
    p.add_argument("--config", required=True, help="ml/configs/*.yaml")
    p.add_argument("--device", default=None, help="e.g. cpu, 0")

    args = parser.parse_args(argv)

    if args.cmd == "attributes":
        from dataprep import attributes

        attributes.run()
    elif args.cmd == "ingest":
        from dataprep import ingest

        ingest.run(
            verify_sha=not args.no_verify_sha, enforce_gates=not args.no_gates
        )
    elif args.cmd == "split":
        from dataprep import grouping, split

        grouping.run()
        split.run(k=args.k)
    elif args.cmd == "tile":
        from dataclasses import replace

        from dataprep import tiler, yolo_convert

        aug = tiler.DEFAULT_AUG
        if args.no_d4:
            aug = replace(aug, d4=False)
        if args.ct_p is not None:
            aug = replace(aug, ct_p=args.ct_p)
        if args.no_handheld_cap:
            aug = replace(aug, handheld_cap=False)
        if args.fixed_k is not None:
            aug = replace(aug, rfs=False, fixed_k=args.fixed_k)
        tiler.run(schema=args.schema, write_images=not args.no_images, aug=aug)
        yolo_convert.run(schema=args.schema, aug=aug)
        if args.zip:
            yolo_convert.export_zip(schema=args.schema, aug=aug)
    elif args.cmd == "qa":
        from dataprep import qa

        qa.run(schema=args.schema, n=args.n)
    elif args.cmd == "report":
        from dataprep import report

        report.run(schema=args.schema)
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
            schema=args.schema,
            bucket=args.bucket,
            operating_point=Path(args.operating_point) if args.operating_point else None,
            run_name=args.run_name,
            conf_floor=args.conf,
        )
    elif args.cmd == "choose-op":
        from eval import run_eval

        run_eval.choose_operating_point(
            args.runs, target_sensitivity=args.target_sensitivity,
            out=Path(args.out) if args.out else None,
        )
    elif args.cmd == "compare":
        from eval import compare

        compare.run(args.run_a, args.run_b)
    elif args.cmd == "import-run":
        import registry

        registry.import_run(Path(args.bundle))
    elif args.cmd == "train":
        from train import ultralytics_run

        ultralytics_run.run(config_path=Path(args.config), device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
