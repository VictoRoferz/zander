"""
zander-ml CLI — run every toolchain step from inside ml/:

    cd ml
    python cli.py attributes        # committed photo-attributes table from the audit
    python cli.py ingest            # manifest from photos + label snapshot + attributes (+gates)
    python cli.py split             # board-aware groups + frozen split_v2
    python cli.py tile --schema def3  # 640-px tiles + offline augmentation + COCO + YOLO view (+--zip)
    python cli.py qa --schema def3  # gallery of augmented training tiles (eyeball once per build)
    python cli.py report            # committed dataset card
    python cli.py train             # one run config -> bundle zip (train.dispatch; local or Colab)
    python cli.py import-run        # bundle zip -> ml_data/bundles/<run>/ + MLflow (zander-v2)
    python cli.py eval-matrix       # import + score every fold bundle on its own fold
    python cli.py eval              # class-aware photo-level evaluation of one ONNX model
    python cli.py log-eval          # an eval run's headline numbers -> its bundle's MLflow run
    python cli.py choose-op         # freeze t* from a family's out-of-fold runs
    python cli.py compare           # paired comparison of two runs or families (G1, NAL gate)
    python cli.py summary           # pooled OOF ranking -> reports/model_selection_v2.md
    python cli.py export            # dev_all bundle + operating point -> detector.onnx + .json
    python cli.py latency           # tiled-inference latency bench for an ONNX model
    python cli.py verifier ...      # crop-verifier prototype (verifier/cli.py)

Steps are idempotent; `split` refuses to overwrite a frozen split file,
`choose-op` a frozen operating point and `export` an existing export (--force
archives). The runbook is ml/README.md.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import settings  # noqa: E402

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ml.cli")

SCHEMA_CHOICES = ["def3", "nal1"]
# Kept in sync with eval.run_eval.BUCKETS (not imported: onnxruntime at parse time).
BUCKET_CHOICES = ["fold0", "fold1", "fold2", "train_always", "test", "probe_xprod", "future"]
# Refusals the user should read as a message, not a traceback.
_USER_ERRORS = {"EvalGuardError", "ExportError", "BundleError", "CompareError"}


def _lazy(module: str, owner: str):
    """Import a module built by another work package, with a clear error if absent."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name and (module == e.name or module.startswith(e.name + ".")):
            raise SystemExit(
                f"`{module}` is not in this checkout yet ({owner}) — merge that work "
                "package first"
            ) from None
        raise


def _overrides(pairs: list[str] | None) -> dict | None:
    """--set a=1 --set extra.fraction=0.05 -> {"a": 1, "extra": {"fraction": 0.05}} (YAML values)."""
    import yaml

    if not pairs:
        return None
    out: dict = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw) if raw else ""
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zander-ml", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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

    p = sub.add_parser("train", help="train one run config -> bundle zip (train.dispatch)")
    p.add_argument("--config", required=True, help="ml/configs/**/*.yaml")
    p.add_argument("--device", default=None, help="e.g. cpu, 0")
    p.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                   help="config override, repeatable; dotted keys nest (extra.fraction=0.05)")
    p.add_argument("--mirror-dir", default=None,
                   help="resume mirror (Colab: MyDrive/ZanderML/mirror)")

    p = sub.add_parser("import-run", help="bundle zip/dir -> ml_data/bundles/<run>/ + MLflow")
    p.add_argument("bundle", help="<run>_bundle.zip or a bundle directory")
    p.add_argument("--force", action="store_true",
                   help="replace a DIFFERENT bundle already imported under the same run name "
                        "(the old one is archived)")

    p = sub.add_parser("eval-matrix", help="import + score every fold bundle on its own fold")
    p.add_argument("--bundles", nargs="+", default=None, metavar="ZIP_OR_GLOB",
                   help="bundle zips or quoted glob patterns (default: "
                        "ml_data/run_dirs/*_bundle.zip)")
    p.add_argument("--re-eval", action="store_true", help="re-score runs that already have metrics")
    p.add_argument("--gallery", action="store_true", help="also write gallery.html (12-17 MB each)")
    p.add_argument("--no-record", action="store_true", help="do not append to reports/results_v2.csv")
    p.add_argument("--force-import", action="store_true", help="import-run --force for every bundle")
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow registration/logging")

    p = sub.add_parser("eval", help="class-aware photo-level evaluation of one model")
    p.add_argument("--model", required=True, help="path to .onnx (meta.json next to it)")
    p.add_argument("--adapter", default=None, choices=["ultralytics", "deim"],
                   help="default: the bundle's meta.json, else ultralytics")
    p.add_argument("--schema", default=None, choices=SCHEMA_CHOICES,
                   help="default: the bundle's meta.json, else def3")
    p.add_argument("--bucket", default=None, choices=BUCKET_CHOICES,
                   help="default: the model's own validation fold (meta.json)")
    p.add_argument("--operating-point", default=None,
                   help="operating_point.json (REQUIRED for test/probe_xprod/future)")
    p.add_argument("--run-name", default=None,
                   help="default: <family>_fold<k> on its own fold, else <bundle_run>__<bucket> "
                        "(+ __look<k> for a further test look; a test look is never overwritten)")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--no-record", action="store_true", help="do not append to reports/results_v2.csv")
    p.add_argument("--no-gallery", action="store_true", help="skip gallery.html")
    p.add_argument("--allow-foreign-fold", action="store_true",
                   help="diagnostics only: score a bucket the model trained on / a model without "
                        "meta.json (stamped guard_overridden, never recorded)")
    p.add_argument("--second-look", default=None, metavar="REASON",
                   help="test bucket: evaluate another model / operating point / --conf / ground "
                        "truth than the recorded looks (reason goes into reports/test_ledger.csv)")

    p = sub.add_parser("log-eval", help="log eval runs into their bundles' MLflow runs")
    p.add_argument("eval_runs", nargs="+")

    p = sub.add_parser("choose-op", help="freeze t* from a family's out-of-fold runs")
    p.add_argument("runs", nargs="*", help="fold eval runs (default: <family>_fold0..2)")
    p.add_argument("--family", default=None, help="family name, e.g. yolo26n_def3")
    p.add_argument("--target-sensitivity", type=float, default=0.95,
                   help="pooled out-of-fold photo sensitivity t* must reach, 0 < x <= 1")
    p.add_argument("--out", default=None,
                   help="default: ml_data/eval/operating_point_<schema>_<family>.json")
    p.add_argument("--force", action="store_true",
                   help="replace a frozen operating point (the old one is archived)")
    p.add_argument("--allow-smoke", action="store_true", help="accept smoke runs (plumbing tests)")

    p = sub.add_parser("compare", help="paired comparison of two eval runs or families")
    p.add_argument("run_a", help="eval run or family (pooled over <family>_fold0..2)")
    p.add_argument("run_b")
    p.add_argument("--class", dest="cls", default="NAL",
                   help="class name, short name (NAL/Brücke/Fahne), run A's index, or any")
    p.add_argument("--margin", type=float, default=None,
                   help="non-inferiority margin for A vs B (the NAL gate uses 0.05)")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("summary", help="pooled OOF ranking, G1 and NAL gate report")
    p.add_argument("--include-smoke", action="store_true")
    p.add_argument("--family", dest="families", action="append", default=None,
                   help="restrict to these families (repeatable)")
    p.add_argument("--out-md", default=None, help="default: ml/reports/model_selection_v2.md")
    p.add_argument("--out-json", default=None, help="default: ml_data/eval/summary_v2.json")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("export", help="verified detector.onnx + detector.json for the inspector")
    p.add_argument("--bundle", required=True, help="imported bundle dir (ml_data/bundles/<run>)")
    p.add_argument("--operating-point", required=True, help="the family's frozen operating point")
    p.add_argument("--eval-run", default=None,
                   help="eval run of this model to validate against (default: its test eval)")
    p.add_argument("--out-dir", default=None, help="default: ml_data/exports")
    p.add_argument("--name", default="detector", help="file stem: <name>.onnx + <name>.json")
    p.add_argument("--allow-fold-model", action="store_true",
                   help="export a fold model instead of the dev_all retrain (recorded)")
    p.add_argument("--verify-photos", type=int, default=5, help="parity photos (0 = skip, recorded)")
    p.add_argument("--no-test-eval", action="store_true",
                   help="export without a test evaluation (recorded as null)")
    p.add_argument("--force", action="store_true", help="archive and replace an existing export")

    p = sub.add_parser("verifier", add_help=False,
                       help="crop-verifier prototype: arguments go to verifier/cli.py")
    p.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def _dispatch(args) -> int:
    if args.cmd == "attributes":
        from dataprep import attributes

        attributes.run()
    elif args.cmd == "ingest":
        from dataprep import ingest

        ingest.run(verify_sha=not args.no_verify_sha, enforce_gates=not args.no_gates)
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
            model_path=Path(args.model), adapter_name=args.adapter,
            n_photos=args.photos, threads=args.threads,
        )
    elif args.cmd == "train":
        dispatch = _lazy("train.dispatch", "WP-train")
        bundle = dispatch.run(
            Path(args.config), device=args.device, overrides=_overrides(args.overrides),
            mirror_dir=Path(args.mirror_dir) if args.mirror_dir else None,
        )
        logger.info(f"bundle: {bundle}")
    elif args.cmd == "import-run":
        import registry

        registry.import_run(Path(args.bundle), force=args.force)
    elif args.cmd == "eval-matrix":
        from eval import matrix

        rows = matrix.run(
            bundles=args.bundles, re_eval=args.re_eval, gallery=args.gallery,
            record=not args.no_record, force_import=args.force_import, mlflow=not args.no_mlflow,
        )
        return 1 if any(r["status"] == "failed" for r in rows) else 0
    elif args.cmd == "eval":
        import registry
        from eval import run_eval

        model = Path(args.model)
        info = registry.bundle_info(model.parent) or {}
        bucket = args.bucket or info.get("fold")
        if not bucket or bucket not in BUCKET_CHOICES:
            raise SystemExit("--bucket is required (no meta.json fold to default to)")
        run_eval.run(
            model_path=model,
            adapter_name=args.adapter or info.get("adapter") or "ultralytics",
            schema=args.schema or info.get("schema") or "def3",
            bucket=bucket,
            operating_point=Path(args.operating_point) if args.operating_point else None,
            run_name=args.run_name,
            conf_floor=args.conf,
            allow_foreign_fold=args.allow_foreign_fold,
            second_look=args.second_look,
            record=not args.no_record,
            gallery=not args.no_gallery,
        )
    elif args.cmd == "log-eval":
        import registry

        missing = [r for r in args.eval_runs if registry.log_eval(r) is None]
        return 1 if missing else 0
    elif args.cmd == "choose-op":
        from eval import run_eval

        run_eval.choose_operating_point(
            args.runs, target_sensitivity=args.target_sensitivity,
            out=Path(args.out) if args.out else None,
            family=args.family, force=args.force, allow_smoke=args.allow_smoke,
        )
    elif args.cmd == "compare":
        from eval import compare

        result = compare.compare(args.run_a, args.run_b, cls=args.cls, margin=args.margin,
                                 seed=args.seed)
        logger.info(compare.summary_line(result))
    elif args.cmd == "summary":
        from eval import summary

        summary.run(
            include_smoke=args.include_smoke, families=args.families,
            out_md=Path(args.out_md) if args.out_md else None,
            out_json=Path(args.out_json) if args.out_json else None, seed=args.seed,
        )
    elif args.cmd == "export":
        import export

        export.run(
            bundle_dir=Path(args.bundle), operating_point=Path(args.operating_point),
            eval_run=args.eval_run,
            out_dir=Path(args.out_dir) if args.out_dir else settings.exports_dir,
            name=args.name, allow_fold_model=args.allow_fold_model,
            verify_photos=args.verify_photos, no_test_eval=args.no_test_eval, force=args.force,
        )
    elif args.cmd == "verifier":
        return int(_lazy("verifier.cli", "WP-verifier").main(list(args.args)) or 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["verifier"]:  # pass everything (incl. --help) through verbatim
        return int(_lazy("verifier.cli", "WP-verifier").main(argv[1:]) or 0)
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except Exception as e:
        if type(e).__name__ in _USER_ERRORS or isinstance(e, FileNotFoundError):
            logger.error(str(e))
            logger.debug("traceback", exc_info=True)  # LOG_LEVEL=DEBUG shows it
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
