"""
Run registry: import portable run_dir bundles into the LOCAL MLflow file
store (a viewer — the bundle stays the source of truth).

    python cli.py import-run ml_data/run_dirs/<name>_bundle.zip

Never merges mlruns/ directories across machines (absolute artifact_uri paths
in meta.yaml corrupt on relocation — known MLflow file-store trap). Instead a
fresh run is created from the bundle's config/meta/metrics; idempotent via a
bundle-sha tag.
"""
from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import zipfile
from pathlib import Path

import yaml

from config.settings import settings

logger = logging.getLogger("ml.registry")

EXPERIMENT = "zander-nal"


def _bundle_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _flatten(prefix: str, obj, out: dict) -> dict:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}{k}." if prefix else f"{k}.", v, out)
    else:
        out[prefix.rstrip(".")] = obj
    return out


def import_run(bundle_path: Path) -> str | None:
    import mlflow

    mlflow.set_tracking_uri(f"file:{settings.mlruns_dir}")
    mlflow.set_experiment(EXPERIMENT)

    if bundle_path.is_file() and bundle_path.suffix == ".zip":
        sha = _bundle_sha(bundle_path)
        tmp = Path(tempfile.mkdtemp(prefix="bundle_"))
        with zipfile.ZipFile(bundle_path) as zf:
            zf.extractall(tmp)
        bundle_dir = tmp
    else:
        bundle_dir = bundle_path
        sha = hashlib.sha256(
            b"".join(
                sorted(
                    p.name.encode() + p.read_bytes()[:1024]
                    for p in bundle_dir.iterdir()
                    if p.is_file()
                )
            )
        ).hexdigest()

    existing = mlflow.search_runs(
        filter_string=f"tags.bundle_sha = '{sha}'", output_format="list"
    )
    if existing:
        logger.info(f"bundle already imported (run {existing[0].info.run_id}) — skipping")
        return existing[0].info.run_id

    config = yaml.safe_load((bundle_dir / "config.yaml").read_text(encoding="utf-8"))
    meta = json.loads((bundle_dir / "meta.json").read_text(encoding="utf-8"))
    metrics = json.loads((bundle_dir / "metrics.json").read_text(encoding="utf-8"))

    with mlflow.start_run(run_name=meta.get("run_name", bundle_path.stem)) as run:
        params = {}
        _flatten("", config, params)
        for key in (
            "fold", "dataset_id", "dataset_fingerprint", "snapshot_id",
            "split_id", "oversample_k", "framework", "git_sha", "git_dirty",
            "gpu", "ultralytics_version", "torch_version",
        ):
            if key in meta and meta[key] is not None:
                params[key] = meta[key]
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        for step, epoch in enumerate(metrics.get("epochs", [])):
            numeric = {
                k.replace("(", "_").replace(")", "_").replace("/", "_"): v
                for k, v in epoch.items()
                if isinstance(v, float)
            }
            if numeric:
                mlflow.log_metrics(numeric, step=step)
        mlflow.set_tags(
            {"bundle_sha": sha, "benchmark": "r1", "model": str(config.get("model"))}
        )
        for f in sorted(bundle_dir.iterdir()):
            if f.is_file() and f.suffix != ".pt":  # weights stay in the bundle
                mlflow.log_artifact(str(f))
        logger.info(
            f"imported as run {run.info.run_id} — view with: "
            f"mlflow ui --backend-store-uri file:{settings.mlruns_dir}"
        )
        return run.info.run_id
