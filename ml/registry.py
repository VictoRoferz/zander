"""
Run registry: bring portable run bundles home, and register them (plus their
photo-level eval results) in the LOCAL MLflow file store — a viewer only; the
extracted bundle and ml_data/eval stay the source of truth.

    python cli.py import-run ../ml_data/run_dirs/<run>_bundle.zip [--force]
    python cli.py log-eval <eval_run> [...]
    mlflow ui --backend-store-uri file:../ml_data/mlruns   # experiment "zander-v2"

Bundles (docs/CONTRACTS.md §8.1):
  - A bundle zip is extracted into ml_data/bundles/<run_name>/ (flat:
    model.onnx, meta.json, config.yaml, ...). Evaluation needs model.onnx and
    meta.json side by side — the own-fold / schema / split guards of
    `cli.py eval` read meta.json.
  - Extraction goes through a staging dir INSIDE ml_data/bundles/ and one
    rename, so a crash never leaves a half bundle or a stray temp dir behind.
  - Identity = sha256 of the zip (a Drive `<zip>.sha256` marker is verified
    when present). The same sha re-imports as a no-op (the model file is
    re-verified); a DIFFERENT sha under an existing run_name is refused unless
    `force` — the old dir is then archived to ml_data/bundles/_archive/,
    never deleted.
  - Bundle v1 (no `bundle_version`) is read too: family derived from run_name.

Never merges mlruns/ directories across machines (absolute artifact_uri paths
in meta.yaml corrupt on relocation — known MLflow file-store trap). Instead a
fresh run is created from the bundle's config/meta/metrics; idempotent via the
bundle_sha tag. Weights (.pt) and ONNX files stay in the bundle dir (tag
`bundle_dir`) instead of being duplicated into mlruns.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from config.settings import REPO_ROOT, settings

logger = logging.getLogger("ml.registry")

EXPERIMENT = "zander-v2"
BUNDLE_MARKER = ".bundle_source.json"
ARCHIVE_DIR = "_archive"
REQUIRED_FILES = ("meta.json", "model.onnx")
NAL = "Nicht ausreichend Lot"
# Files that stay in the bundle dir only (large binaries).
_NOT_LOGGED_SUFFIXES = (".pt", ".pth", ".onnx")
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_FAMILY_SUFFIX_RE = re.compile(r"_(fold\d+|devall|dev_all)$")


class BundleError(ValueError):
    """The bundle is malformed, unsafe, or conflicts with an imported one."""


# ------------------------------------------------------------------ hashing


def _bundle_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def file_sha256(path: Path) -> str:
    """sha256 of a file, memoised on (path, size, mtime) — ONNX files are big
    and the eval/summary code asks for the same model's sha many times."""
    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = _bundle_sha(path)
    return _SHA_CACHE[key]


def _dir_sha(d: Path) -> str:
    """Content hash of a flat bundle directory (file names + file hashes)."""
    h = hashlib.sha256()
    for p in sorted(d.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            h.update(p.name.encode("utf-8") + b"\0" + file_sha256(p).encode() + b"\n")
    return h.hexdigest()


# ------------------------------------------------------------ bundle reading


def family_from_run_name(run_name: str) -> str:
    """`yolo26n_def3_fold0` / `yolo26n_def3_devall` -> `yolo26n_def3`
    (CONTRACTS §8.1); names without a fold suffix are their own family."""
    return _FAMILY_SUFFIX_RE.sub("", run_name)


def normalize_fold(fold) -> str | None:
    if fold is None or fold == "":
        return None
    fold = str(fold)
    if fold in ("devall", "dev-all"):
        return "dev_all"
    if fold.isdigit():
        return f"fold{fold}"
    return fold


def _schema_from(meta: dict, config: dict) -> str | None:
    from dataprep.remap import SCHEMAS

    if meta.get("schema"):
        return meta["schema"]
    classes = meta.get("classes")
    for name, cls in SCHEMAS.items():
        if classes == cls:
            return name
    ds = str(meta.get("dataset_id") or config.get("dataset_id") or "")
    for name in SCHEMAS:
        if ds.endswith(f"_{name}") or f"_{name}_" in ds:
            return name
    return None


def resolve_model_path(model: str | os.PathLike | None) -> Path | None:
    """A `model` string recorded by an eval run -> an existing file, or None.
    v1 runs recorded paths relative to ml/ (the CLI's working directory)."""
    if not model:
        return None
    p = Path(model)
    candidates = [p] if p.is_absolute() else [REPO_ROOT / "ml" / p, Path.cwd() / p]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def bundle_info(bundle_dir: Path) -> dict | None:
    """Normalised view of a bundle dir's meta.json + config.yaml (v1 or v2),
    or None if there is no meta.json."""
    bundle_dir = Path(bundle_dir)
    meta_path = bundle_dir / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg_path = bundle_dir / "config.yaml"
    config = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}) if cfg_path.is_file() else {}
    run_name = meta.get("run_name") or config.get("name") or bundle_dir.name
    weights = meta.get("weights") or config.get("final_weights")
    if weights is None:
        weights = "last" if (bundle_dir / "last.pt").exists() else (
            "best" if (bundle_dir / "best.pt").exists() else None)
    adapter = meta.get("adapter") or config.get("adapter") or (
        "deim" if str(meta.get("framework", "")).lower().startswith("deim") else
        "ultralytics" if meta.get("framework") == "ultralytics" else None)
    onnx = meta.get("onnx") if isinstance(meta.get("onnx"), dict) else {}
    marker = _read_marker(bundle_dir)
    model = bundle_dir / "model.onnx"
    return {
        "bundle_dir": bundle_dir,
        "bundle_version": int(meta.get("bundle_version") or 1),
        "run_name": run_name,
        "family": meta.get("family") or family_from_run_name(run_name),
        "fold": normalize_fold(meta.get("fold") or config.get("fold")),
        "schema": _schema_from(meta, config),
        "adapter": adapter,
        "classes": meta.get("classes"),
        "split_id": meta.get("split_id"),
        "dataset_id": meta.get("dataset_id") or config.get("dataset_id"),
        "dataset_fingerprint": meta.get("dataset_fingerprint"),
        "snapshot_id": meta.get("snapshot_id"),
        "tiler_version": meta.get("tiler_version"),
        "framework": meta.get("framework"),
        "weights": weights,
        "epochs": meta.get("epochs") or config.get("epochs"),
        "config": config,
        "config_sha256": meta.get("config_sha256") or (
            _bundle_sha(cfg_path) if cfg_path.is_file() else None),
        "onnx_layout": onnx.get("layout"),
        "model_path": model if model.is_file() else None,
        "bundle_sha256": (marker or {}).get("bundle_sha256"),
        "git_sha": meta.get("git_sha"),
        "versions": meta.get("versions") or {
            k: meta[k] for k in ("ultralytics_version", "torch_version", "deim_commit") if meta.get(k)
        },
        "meta": meta,
    }


def is_smoke(name: str | None) -> bool:
    """Smoke runs (plumbing checks, not models) never enter benchmarks."""
    return bool(name) and "smoke" in str(name).lower()


# ------------------------------------------------------------------- import


def _read_marker(bundle_dir: Path) -> dict | None:
    p = Path(bundle_dir) / BUNDLE_MARKER
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _verify_sha_marker(zip_path: Path, sha: str) -> None:
    """A Drive `<zip>.sha256` (or `<stem>.sha256`) marker must match."""
    for marker in (zip_path.with_name(zip_path.name + ".sha256"), zip_path.with_suffix(".sha256")):
        if marker.is_file():
            tokens = marker.read_text(encoding="utf-8").split()
            expected = tokens[0].lower() if tokens else ""
            if expected != sha:
                raise BundleError(
                    f"{zip_path.name}: sha256 {sha[:12]}… does not match its marker "
                    f"{marker.name} ({expected[:12]}…) — incomplete or corrupt download"
                )
            return


def _safe_member(name: str) -> bool:
    return bool(name) and "/" not in name and "\\" not in name and name not in (".", "..") \
        and not name.startswith(".") and ":" not in name


def _zip_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = [m for m in zf.infolist() if not m.is_dir()]
    bad = [m.filename for m in members if not _safe_member(m.filename)]
    if bad:
        raise BundleError(f"bundle zips are flat — refusing unsafe/nested members {bad[:5]}")
    names = {m.filename for m in members}
    missing = [f for f in REQUIRED_FILES if f not in names]
    if missing:
        raise BundleError(f"bundle is missing {missing}")
    return members


def _run_name_of(meta: dict, source: Path) -> str:
    stem = source.stem[: -len("_bundle")] if source.stem.endswith("_bundle") else source.stem
    name = meta.get("run_name") or stem
    if not _RUN_NAME_RE.match(name) or name == ARCHIVE_DIR:
        raise BundleError(f"unsafe run_name {name!r} in {source.name}")
    return name


def _write_marker(bundle_dir: Path, doc: dict) -> None:
    (bundle_dir / BUNDLE_MARKER).write_text(json.dumps(doc, indent=1), encoding="utf-8")


def _place(target: Path, sha: str, force: bool, fill, source: str) -> str:
    """Put a bundle at `target` via a staging dir + rename. Returns the status."""
    status = "imported"
    if target.exists():
        marker = _read_marker(target)
        model = target / "model.onnx"
        if (
            marker and marker.get("bundle_sha256") == sha and model.is_file()
            and (target / "meta.json").is_file()
            and marker.get("files", {}).get("model.onnx") in (None, file_sha256(model))
        ):
            return "present"
        old = (marker or {}).get("bundle_sha256") or "unknown"
        if not force:
            raise BundleError(
                f"{target} already holds a different bundle (sha {old[:12]}… vs "
                f"{sha[:12]}…, or its files changed) — pass --force to archive it "
                "and import this one"
            )
        archive = settings.bundles_dir / ARCHIVE_DIR / (
            f"{target.name}__{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(target, archive)
        logger.warning(f"--force: archived the previous {target.name} bundle to {archive}")
        status = "replaced"
    staging = Path(tempfile.mkdtemp(prefix=f".staging-{target.name}-", dir=settings.bundles_dir))
    try:
        fill(staging)
        missing = [f for f in REQUIRED_FILES if not (staging / f).is_file()]
        if missing:
            raise BundleError(f"bundle is missing {missing}")
        _write_marker(staging, {
            "bundle_sha256": sha,
            "source": source,
            "imported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "files": {p.name: file_sha256(p) for p in sorted(staging.iterdir())
                      if p.is_file() and not p.name.startswith(".")},
        })
        os.chmod(staging, 0o755)  # mkdtemp is 0700; bundles are plain readable data
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return status


def _sweep_stale_staging(max_age_s: float = 3600.0) -> None:
    """Staging dirs survive only a hard kill mid-import; they are never valid state."""
    now = datetime.now(timezone.utc).timestamp()
    for d in settings.bundles_dir.glob(".staging-*"):
        try:
            if d.is_dir() and now - d.stat().st_mtime > max_age_s:
                shutil.rmtree(d, ignore_errors=True)
                logger.info(f"removed stale staging dir {d.name}")
        except OSError:
            pass


def _extract_zip(zip_path: Path, members: list[zipfile.ZipInfo], dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for m in members:
            out = dest / m.filename
            with zf.open(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            os.chmod(out, 0o644)


def _copy_dir(src_dir: Path, dest: Path) -> None:
    for p in sorted(src_dir.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            shutil.copyfile(p, dest / p.name)


def import_bundle(bundle_path: Path, force: bool = False, register: bool = True) -> dict:
    """Extract/copy a bundle into ml_data/bundles/<run_name>/ and register it.

    Returns {run_id, run_name, bundle_dir, bundle_sha256, status, info} with
    status one of imported | present | replaced | registered-in-place.
    """
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"bundle not found: {bundle_path}")
    settings.bundles_dir.mkdir(parents=True, exist_ok=True)
    _sweep_stale_staging()

    if bundle_path.is_file():
        if bundle_path.suffix != ".zip":
            raise BundleError(f"expected a <run>_bundle.zip or a bundle dir, got {bundle_path.name}")
        sha = _bundle_sha(bundle_path)
        _verify_sha_marker(bundle_path, sha)
        with zipfile.ZipFile(bundle_path) as zf:
            members = _zip_members(zf)
            meta = json.loads(zf.read("meta.json").decode("utf-8"))
        run_name = _run_name_of(meta, bundle_path)
        target = settings.bundles_dir / run_name
        status = _place(target, sha, force,
                        lambda d: _extract_zip(bundle_path, members, d), str(bundle_path))
    else:
        marker = _read_marker(bundle_path)
        in_store = bundle_path.resolve().parent == settings.bundles_dir.resolve()
        if in_store and marker:
            target, sha, status = bundle_path, marker["bundle_sha256"], "registered-in-place"
            run_name = bundle_path.name
        elif in_store:  # placed by hand: adopt it where it is (never copy onto itself)
            missing = [f for f in REQUIRED_FILES if not (bundle_path / f).is_file()]
            if missing:
                raise BundleError(f"{bundle_path} is missing {missing}")
            target, sha, status = bundle_path, _dir_sha(bundle_path), "adopted"
            run_name = bundle_path.name
            _write_marker(bundle_path, {
                "bundle_sha256": sha, "source": str(bundle_path),
                "imported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "files": {p.name: file_sha256(p) for p in sorted(bundle_path.iterdir())
                          if p.is_file() and not p.name.startswith(".")},
            })
        else:
            meta_path = bundle_path / "meta.json"
            if not meta_path.is_file():
                raise BundleError(f"{bundle_path} has no meta.json — not a bundle dir")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sha = _dir_sha(bundle_path)
            run_name = _run_name_of(meta, bundle_path)
            target = settings.bundles_dir / run_name
            status = _place(target, sha, force, lambda d: _copy_dir(bundle_path, d), str(bundle_path))

    info = bundle_info(target)
    logger.info(f"bundle {run_name}: {status} -> {target}")
    run_id = _register(target, info, sha) if register else None
    return {"run_id": run_id, "run_name": run_name, "bundle_dir": target,
            "bundle_sha256": sha, "status": status, "info": info}


def import_run(bundle_path: Path, force: bool = False) -> str | None:
    """Backward-compatible entry point: import a bundle, return the MLflow run id."""
    return import_bundle(Path(bundle_path), force=force)["run_id"]


# ------------------------------------------------------------------- MLflow


def _flatten(prefix: str, obj, out: dict) -> dict:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}{k}." if prefix else f"{k}.", v, out)
    else:
        out[prefix.rstrip(".")] = obj
    return out


def _mlflow():
    import mlflow

    mlflow.set_tracking_uri(f"file:{settings.mlruns_dir}")
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def _search(mlflow, **tags) -> list:
    clauses = [f"tags.{k} = '{v}'" for k, v in tags.items() if v]
    if not clauses:
        return []
    return mlflow.search_runs(
        experiment_names=[EXPERIMENT], filter_string=" and ".join(clauses),
        output_format="list",
    )


def _register(bundle_dir: Path, info: dict, sha: str) -> str:
    mlflow = _mlflow()
    existing = _search(mlflow, bundle_sha=sha)
    if existing:
        logger.info(f"bundle already registered (MLflow run {existing[0].info.run_id}) — skipping")
        return existing[0].info.run_id

    model_sha = file_sha256(bundle_dir / "model.onnx")
    for old in _search(mlflow, run_name=info["run_name"]):
        if old.data.tags.get("superseded_by") is None:
            mlflow.MlflowClient().set_tag(old.info.run_id, "superseded_by", sha)

    config, meta = info["config"], info["meta"]
    metrics_path = bundle_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    with mlflow.start_run(run_name=info["run_name"]) as run:
        params: dict = {}
        _flatten("", config, params)
        for key in (
            "fold", "dataset_id", "dataset_fingerprint", "snapshot_id", "split_id",
            "schema", "tiler_version", "framework", "git_sha", "git_dirty", "gpu",
            "weights", "epochs_completed", "train_seconds", "config_sha256",
            "ultralytics_version", "torch_version",
        ):
            if meta.get(key) is not None and key not in params:
                params[key] = meta[key]
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        for step, epoch in enumerate(metrics.get("epochs", []) if isinstance(metrics, dict) else []):
            numeric = {
                k.replace("(", "_").replace(")", "_").replace("/", "_"): v
                for k, v in epoch.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if numeric:
                mlflow.log_metrics(numeric, step=step)
        mlflow.set_tags({
            "run_name": info["run_name"], "family": info["family"],
            "schema": str(info["schema"]), "fold": str(info["fold"]),
            "split_id": str(info["split_id"]), "benchmark": str(info["split_id"]),
            "bundle_sha": sha, "model_sha256": model_sha,
            "model": str(config.get("model") or meta.get("model")),
            "adapter": str(info["adapter"]), "bundle_version": str(info["bundle_version"]),
            "bundle_dir": str(bundle_dir), "smoke": str(is_smoke(info["run_name"])).lower(),
        })
        for f in sorted(bundle_dir.iterdir()):
            if f.is_file() and not f.name.startswith(".") and f.suffix not in _NOT_LOGGED_SUFFIXES:
                mlflow.log_artifact(str(f))
        logger.info(
            f"registered as MLflow run {run.info.run_id} (experiment {EXPERIMENT}) — view with: "
            f"mlflow ui --backend-store-uri file:{settings.mlruns_dir}"
        )
        return run.info.run_id


def eval_key_metrics(m: dict) -> dict:
    """The headline numbers of one eval run's metrics.json (None when absent)."""
    l1 = m.get("level1_verdict") or {}
    nal = ((m.get("level3_defects") or {}).get("per_class") or {}).get(NAL) or {}
    has_nal = bool(nal.get("n_gt"))
    return {
        "recall_nal_0.5fppi": nal.get("recall_at_op") if has_nal else None,
        "escape": (l1.get("escape") or {}).get("rate"),
        "fcr_seated": (l1.get("false_call_seated") or {}).get("rate"),
        "photo_auroc": l1.get("photo_auroc"),
        "ap50_nal": nal.get("ap50") if has_nal else None,
        "ms_per_photo": m.get("mean_ms_per_photo"),
    }


def find_mlflow_run(model_sha256: str | None = None, run_name: str | None = None):
    """The (non-superseded) MLflow run of a model: by ONNX sha, else by run name."""
    mlflow = _mlflow()
    for tags in ({"model_sha256": model_sha256}, {"run_name": run_name}):
        runs = [r for r in _search(mlflow, **tags) if "superseded_by" not in r.data.tags]
        if runs:
            return runs[0]
    return None


def log_eval(eval_run: str) -> str | None:
    """Log one eval run's headline numbers + report into its bundle's MLflow run
    as eval.<bucket>.<metric>. Returns the MLflow run id (None if the bundle was
    never imported)."""
    run_dir = settings.eval_dir / eval_run
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    src = metrics.get("source") or {}
    model_sha = src.get("model_sha256")
    if not model_sha:
        model = resolve_model_path(metrics.get("model"))
        model_sha = file_sha256(model) if model else None
    run = find_mlflow_run(model_sha, src.get("bundle_run"))
    if run is None:
        logger.warning(
            f"log-eval {eval_run}: no MLflow run for this model (sha "
            f"{(model_sha or '?')[:12]}…) — import its bundle first (`cli.py import-run`)"
        )
        return None
    mlflow = _mlflow()
    bucket = metrics["bucket"]
    values = {f"eval.{bucket}.{k}": float(v) for k, v in eval_key_metrics(metrics).items()
              if v is not None}
    with mlflow.start_run(run_id=run.info.run_id):
        if values:
            mlflow.log_metrics(values)
        mlflow.set_tag(f"eval.{bucket}.run", eval_run)
        for name in ("report.md", "metrics.json"):
            if (run_dir / name).is_file():
                mlflow.log_artifact(str(run_dir / name), artifact_path=f"eval/{eval_run}")
    logger.info(f"logged eval {eval_run} into MLflow run {run.info.run_id}: {values}")
    return run.info.run_id
