"""
Cascade evaluation runs: detector + verifier, written as ordinary eval runs
through the FROZEN eval API (run_eval._write_outputs, photo_scorer.score_all)
so every existing tool (load_predictions, compare, choose-op, reports) reads
them. predictions.json is v2 (CONTRACTS §8.2): `scores` = the anchored final
scores the decision uses, `det_scores` = the detector scores of the same
candidates, `source.kind = "cascade"`, per-photo width/height.

    python verifier/cli.py cascade-eval --run-name v1 --model <detector.onnx> \
        --schema def3 --bucket fold0 [--operating-point op.json]

- The cascade's decision threshold is the stacker's anchor t* by construction;
  buckets that need a frozen threshold (test / probe_xprod / future) also
  require --operating-point, whose threshold must equal that t*.
- The test bucket is refused until run_eval.check_test_ledger exists
  (WP-eval), and for any verifier whose exit decision is not "ship". The
  ledger key of a cascade is sha256("cascade:<detector sha>:<stacker sha>"), so
  evaluating the cascade after the detector alone is a second look
  (--second-look "<reason>", recorded).
- Cross-fit runs are written without a results row or gallery; cascade-eval
  runs record a row unless the verifier is a smoke run.
"""
from __future__ import annotations

import contextlib
import inspect
import json
import logging
import platform
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import settings
from dataprep.ingest import load_manifest, photo_path
from dataprep.split import load_split
from eval import run_eval
from eval.photo_scorer import score_all
from infer.pipeline import TiledDetector, load_photo_rgb
from infer.verifier import Verifier
from verifier.common import (
    STACKER, Refusal, read_json, sha256_file, vrun_dir, write_bytes_atomic,
)

logger = logging.getLogger("ml.verifier.cascade")

REQUIRED_METRICS = (
    "run_name", "model", "adapter", "schema", "bucket", "threshold", "threshold_source",
    "n_photos", "mean_ms_per_photo", "tiler_version", "split_id", "snapshot_id",
    "dataset_fingerprint", "host",
)


@contextlib.contextmanager
def legacy_writer(record: bool = True, gallery: bool = True):
    """Around a pre-v2 run_eval._write_outputs (no record/gallery keywords):
    no-op the results row and/or gallery for this call and, as a safety net,
    restore ml/reports/results_v2.csv if it changed although record=False."""
    path = settings.reports_dir / run_eval.RESULTS_FILE
    before = path.read_bytes() if path.exists() else None
    patched = {}
    if not record and hasattr(run_eval, "_append_results_row"):
        patched["_append_results_row"] = run_eval._append_results_row
        run_eval._append_results_row = lambda m: logger.info(
            "results_v2.csv row suppressed (%s)", m.get("run_name"))
    if not gallery and hasattr(run_eval, "_gallery"):
        patched["_gallery"] = run_eval._gallery
        run_eval._gallery = lambda *a, **k: None
    try:
        yield
    finally:
        for name, fn in patched.items():
            setattr(run_eval, name, fn)
        if not record:
            after = path.read_bytes() if path.exists() else None
            if after != before:
                if before is None:
                    path.unlink()
                else:
                    write_bytes_atomic(path, before)
                logger.warning("reverted an unexpected change to %s", path)


def _write_outputs(out_dir: Path, metrics: dict, photos: list[dict], *, record: bool,
                   gallery: bool, source: dict) -> None:
    """run_eval's writer: the v2 one (record/gallery/source keywords, metrics.json
    written last) when present, else the legacy frozen one."""
    writer = getattr(run_eval, "write_outputs", None) or run_eval._write_outputs
    params = inspect.signature(writer).parameters
    if all(k in params for k in ("record", "gallery", "source")):
        writer(out_dir, metrics, photos, record=record, gallery=gallery, source=source)
        return
    with legacy_writer(record=record, gallery=gallery):
        run_eval._write_outputs(out_dir, metrics, photos)


def ensure_predictions_v2(out_dir: Path, photos: list[dict], source: dict) -> bool:
    """Complete predictions.json to v2 whichever writer produced it; only adds
    what is missing (a v2 writer's output is left untouched). True if changed."""
    path = out_dir / "predictions.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    if doc.get("format_version") != 2:
        doc["format_version"] = 2
        changed = True
    src = doc.get("source") or {}
    if src.get("kind") != "cascade":
        doc["source"] = dict(source)
        changed = True
    else:
        for key, value in source.items():
            if src.get(key) is None and value is not None:
                src[key] = value
                changed = True
    if len(doc["photos"]) != len(photos):
        raise RuntimeError("predictions.json photo count differs from the scored photos")
    for rec, p in zip(doc["photos"], photos):
        if rec["capture_id"] != p["capture_id"]:
            raise RuntimeError("predictions.json photo order differs from the scored photos")
        for key in ("width", "height"):
            if rec.get(key) is None:
                rec[key] = p[key]
                changed = True
        if rec.get("det_scores") is None:
            rec["det_scores"] = [float(v) for v in p["det_scores"]]
            changed = True
        if len(rec["det_scores"]) != len(rec["scores"]):
            raise RuntimeError("det_scores and scores differ in length")
    if changed:
        write_bytes_atomic(path, json.dumps(doc, ensure_ascii=False).encode("utf-8"))
    return changed


def write_run(
    run_name: str,
    photos: list[dict],
    class_names: list[str],
    bucket: str,
    t_star: float,
    threshold_source: str,
    metrics_base: dict,
    source: dict,
    smoke: bool,
    probe: Optional[list[dict]] = None,
    record: bool = False,
    gallery: bool = False,
) -> dict:
    """score_all + run_eval's writer + v2 completion + reload check. Smoke runs
    never record a results_v2.csv row, whatever `record` says."""
    metrics = score_all(photos, class_names, threshold=t_star,
                        threshold_source=threshold_source, probe=probe)
    metrics.update(metrics_base)
    metrics.update({"run_name": run_name, "bucket": bucket, "n_photos": len(photos),
                    "source": source, "smoke": smoke})
    missing = [k for k in REQUIRED_METRICS if k not in metrics]
    if missing:
        raise KeyError(f"cascade metrics lack {missing}")
    out_dir = settings.eval_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_outputs(out_dir, metrics, photos, record=record and not smoke, gallery=gallery,
                   source=source)
    if ensure_predictions_v2(out_dir, photos, source):
        # keep metrics.json the last file written: it marks a complete run
        mpath = out_dir / "metrics.json"
        write_bytes_atomic(mpath, mpath.read_bytes())
    # the DoD check: the run loads through the frozen API and scores again
    back = run_eval.load_predictions(run_name)
    again = score_all(back["photos"], back["class_names"], threshold=t_star,
                      threshold_source="reload check")
    if again["level1_verdict"]["n_positive"] != metrics["level1_verdict"]["n_positive"]:
        raise RuntimeError(f"{run_name}: reloaded predictions score differently")
    logger.info(f"cascade eval run written: {out_dir}")
    return metrics


# ------------------------------------------------------------------ cascade-eval


def _bundle_run_name(model_path: Path) -> str:
    meta = model_path.parent / "meta.json"
    if meta.exists():
        return read_json(meta).get("run_name") or model_path.stem
    return model_path.stem


def cascade_identity(model_sha: str, stacker_sha: str) -> str:
    """One sha for the whole decision system (detector + stacker.json, which pins
    verifier.onnx): the test ledger's model key for a cascade."""
    import hashlib

    return hashlib.sha256(f"cascade:{model_sha}:{stacker_sha}".encode()).hexdigest()


def _call_with_known(fn, known: dict):
    """Call fn with the parameters it declares, taken from `known`; refuse if it
    needs one we cannot supply (keeps working across run_eval versions)."""
    kwargs = {}
    for name, prm in inspect.signature(fn).parameters.items():
        if prm.kind in (prm.VAR_POSITIONAL, prm.VAR_KEYWORD):
            continue
        if name in known:
            kwargs[name] = known[name]
        elif prm.default is inspect.Parameter.empty:
            raise RuntimeError(
                f"run_eval.{fn.__name__} needs {name!r}, which cascade-eval cannot supply "
                "— wire it before evaluating the cascade on test"
            )
    return fn(**kwargs)


def check_test_bucket(decision: Optional[str], smoke: bool, **known):
    """The one-shot test bucket: only a shipped, non-smoke cascade, only through
    the test ledger (run_eval.check_test_ledger, WP-eval). Returns the ledger's
    look for record_test_look."""
    fn = getattr(run_eval, "check_test_ledger", None)
    if fn is None:
        raise Refusal(
            "test bucket refused: run_eval.check_test_ledger does not exist yet — the "
            "one-shot test ledger (WP-eval) must be in place before any cascade touches test"
        )
    if smoke or decision != "ship":
        raise Refusal(
            f"test bucket refused: the verifier's exit decision is {decision!r} — only a "
            "pre-registered 'ship' decision earns the cascade a look at the test set"
        )
    return _call_with_known(fn, known)


def run(
    vrun: str,
    model_path: Path,
    adapter: str = "ultralytics",
    schema: str = "def3",
    bucket: str = "fold0",
    operating_point: Optional[Path] = None,
    run_name: Optional[str] = None,
    threads: int = 0,
    second_look: Optional[str] = None,
) -> dict:
    stacker_path = vrun_dir(vrun) / STACKER
    ver = Verifier(stacker_path, threads=threads)
    sdoc = ver.stacker.doc
    if sdoc.get("schema") != schema:
        raise ValueError(f"verifier {vrun} is for schema {sdoc.get('schema')!r}, not {schema!r}")
    prov = sdoc.get("provenance") or {}
    smoke = bool(prov.get("smoke")) or ver.stacker.exit_decision == "invalid-smoke"
    t_star = ver.stacker.t_star
    model_path = Path(model_path)
    bundle_run = _bundle_run_name(model_path)
    run_name = run_name or f"{bundle_run}+{vrun}__{bucket}"
    model_sha = sha256_file(model_path)
    stacker_sha = sha256_file(stacker_path)
    if bucket in run_eval.FROZEN_OP_BUCKETS and operating_point is None:
        raise Refusal(f"bucket {bucket!r} needs --operating-point (frozen threshold)")
    look, ledger = None, {}
    if bucket == "test":
        ledger = {"model_sha": cascade_identity(model_sha, stacker_sha),
                  "op_sha": sha256_file(operating_point), "second_look": second_look,
                  "run_name": run_name, "bundle_run": bundle_run, "family": prov.get("family"),
                  "schema": schema, "threshold": t_star}
        look = check_test_bucket(ver.stacker.exit_decision, smoke, **ledger)
    threshold_source = f"stacker anchor t* ({stacker_path.name}, verifier {vrun})"
    if operating_point is not None:
        op = read_json(operating_point)
        if op.get("schema") != schema:
            raise ValueError(f"operating point is for schema {op.get('schema')!r}")
        if abs(float(op["threshold"]) - t_star) > 1e-12:
            raise Refusal(
                f"operating point t={op['threshold']} differs from the stacker's anchor "
                f"t*={t_star}: the cascade's scores are anchored to t*"
            )
        threshold_source = f"frozen: {operating_point} (= stacker anchor t*)"
    if smoke:
        logger.warning("verifier %s is a SMOKE run: numbers are meaningless", vrun)

    split = load_split()
    manifest = load_manifest()
    entries, probe = run_eval.bucket_photos(bucket, schema)
    detector = TiledDetector(model_path, adapter, conf_floor=ver.stacker.candidate_floor)
    n_model = getattr(detector.adapter, "num_classes", None)
    if n_model is not None and n_model != len(ver.class_names):
        raise ValueError(f"model outputs {n_model} classes, the verifier has {len(ver.class_names)}")
    photos, det_ms, ver_ms = [], [], []
    for n, e in enumerate(entries, 1):
        rgb = load_photo_rgb(photo_path(e))
        det = detector.detect(rgb)
        t0 = time.perf_counter()
        r = ver.score_candidates(rgb, det.boxes, det.scores, det.labels)
        ver_ms.append((time.perf_counter() - t0) * 1000.0)
        det_ms.append(det.elapsed_ms)
        order = np.argsort(-r["final"], kind="stable")
        photos.append({
            **run_eval._gt_record(e, schema, split),
            "boxes": [r["boxes"][i] for i in order],
            "scores": [float(r["final"][i]) for i in order],
            "labels": [int(r["labels"][i]) for i in order],
            "det_scores": [float(r["det"][i]) for i in order],
            "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
        })
        if n % 10 == 0 or n == len(entries):
            logger.info(f"cascade-evaluated {n}/{len(entries)} photos")
    meta = model_path.parent / "meta.json"
    meta_doc = read_json(meta) if meta.exists() else {}
    source = {
        "kind": "cascade", "bundle_run": bundle_run, "family": meta_doc.get("family") or prov.get("family"),
        "fold": meta_doc.get("fold"), "model_sha256": model_sha,
        "model_resolved": str(model_path.resolve()), "adapter": adapter,
        "conf_floor": ver.stacker.candidate_floor, "verifier_run": vrun,
        "stacker_sha256": stacker_sha,
    }
    base = {
        "model": f"{model_path}+{vrun}", "adapter": adapter, "schema": schema,
        "mean_ms_per_photo": round(float(np.mean(np.add(det_ms, ver_ms))), 1) if photos else None,
        "mean_ms_detector": round(float(np.mean(det_ms)), 1) if photos else None,
        "mean_ms_verifier": round(float(np.mean(ver_ms)), 1) if photos else None,
        "tiler_version": detector.tiler_version, "split_id": split["split_id"],
        "snapshot_id": manifest["snapshot_id"],
        "dataset_fingerprint": meta_doc.get("dataset_fingerprint"), "host": platform.node(),
    }
    metrics = write_run(run_name, photos, ver.class_names, bucket, t_star, threshold_source,
                        base, source, smoke, probe=probe, record=not smoke, gallery=True)
    if look is not None and hasattr(run_eval, "record_test_look"):
        _call_with_known(run_eval.record_test_look, {"look": look, **ledger})
    return metrics
