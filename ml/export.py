"""
Export the chosen detector for the inspector: ONNX + detector.json card
(docs/CONTRACTS.md §1) — the only way a model reaches production.

    python cli.py export --bundle ../ml_data/bundles/yolo26n_def3_devall \\
        --operating-point ../ml_data/eval/operating_point_def3_yolo26n_def3.json
    # -> ml_data/exports/detector.onnx + detector.json; copy BOTH to DATA_ROOT/models/

Checks, all before anything is written:
  1. bundle meta vs operating point: schema, class names, split, tiler
     version, and run_eval.op_model_problems — the identity check a test eval
     runs: same family; the op's fold bundles are imported and still hold the
     models it was frozen from (else a stale operating point); their training
     config is equivalent to this bundle's (run_eval.EQUIV_KEYS: arch,
     dataset_id, epochs, imgsz, batch — meta.batch_effective when the trainer
     changed it —, augment, final_weights, `extra` minus NON_TRAINING_EXTRA;
     plus the dataset fingerprint);
  2. the bundle is the dev_all retrain (`allow_fold_model` records an exception;
     a v1 operating point, which predates these guards, only with it);
  3. a TEST evaluation of exactly this model (sha256) at exactly this operating
     point (the run's recorded operating_point.sha256) exists (`no_test_eval`
     records null instead); only detector runs count, never a cascade run of
     the same detector. Further looks live in `<bundle_run>__test__look<k>`:
     of several matching looks the latest (metrics.json written last) is
     taken; looks at other operating points never count;
  4. parity: TiledDetector(conf_floor=t) + infer.verdict.photo_verdict on
     `verify_photos` photos must reproduce the verdict (N.iO/i.O + class set)
     that the reference eval's predictions.json implies at t — zero mismatches.
     Photos nearest the threshold are chosen (that is where parity breaks).
     Without any eval of this model it compares floor t against floor 0.001
     live instead.
Then the ONNX is copied atomically as <name>.onnx and the card is written with
infer.detector_card.write_detector_card (the inspector runs the same
validator) and read back. An existing export is refused unless `force`, which
archives it to <out_dir>/archive/<timestamp>/.

conf_floor = threshold is safe without a verifier (CONTRACTS §1): merging is
greedy by descending score and a kept box keeps its own score, so every
detection >= t — hence the verdict and the class set — is identical at floor t
and at 0.001 (only outlines are tighter, and `photo_score` of an i.O photo
becomes 0 because sub-threshold boxes are no longer produced).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import registry
from config.settings import settings
from dataprep.remap import schema_classes
from dataprep.split import load_split
from infer.detector_card import (
    KNOWN_ADAPTERS,
    card_path_for,
    load_detector_card,
    write_detector_card,
)
from infer.tiling import STRIDE, TILE_SIZE, TILER_VERSION
from infer.verdict import photo_score, photo_verdict

logger = logging.getLogger("ml.export")

DECISION_RULE = (
    "NiO iff any detection score >= threshold (inclusive); classes = those with >= 1 such detection"
)
# v1 predictions.json stored scores rounded to 5 decimals: a live score this
# close to t may legitimately land on the other side of it.
V1_SCORE_TOLERANCE = 5e-6
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ExportError(RuntimeError):
    """The export would ship an unverified or inconsistent model."""


# ------------------------------------------------------------------ helpers


def _photo_paths(capture_ids: list[str]) -> dict[str, Path]:
    """capture_id -> photo file (manifest); tests monkeypatch this."""
    from dataprep.ingest import load_manifest, photo_path

    wanted = set(capture_ids)
    return {e["capture_id"]: photo_path(e) for e in load_manifest()["entries"]
            if e["capture_id"] in wanted}


def _dev_photo_ids(n: int) -> list[str]:
    """Deterministic usable fold photos (never test) for the live floor check."""
    from dataprep.ingest import load_manifest, usable
    from dataprep.split import bucket_of

    split = load_split()
    ids = sorted(e["capture_id"] for e in load_manifest()["entries"]
                 if usable(e) and bucket_of(e, split).startswith("fold"))
    return ids[: max(n * 3, n)]


def _eval_runs_of_model(model_sha: str) -> list[tuple[str, dict]]:
    """(eval run, metrics) of every DETECTOR eval run of exactly this model file.

    A cascade run (detector + verifier, `source.kind = "cascade"`) records the
    detector's sha too, but its scores and verdicts are the cascade's: it is
    neither this model's test evaluation nor a parity reference. A run without
    `source.kind` (v1) is a detector run."""
    out = []
    if not settings.eval_dir.is_dir():
        return out
    for d in sorted(settings.eval_dir.iterdir()):
        mpath = d / "metrics.json"
        if not mpath.is_file() or not (d / "predictions.json").is_file():
            continue
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if ((m.get("source") or {}).get("kind") or "detector") != "detector":
            continue
        sha = (m.get("source") or {}).get("model_sha256")
        if not sha:
            model = registry.resolve_model_path(m.get("model"))
            sha = registry.file_sha256(model) if model else None
        if sha == model_sha:
            out.append((d.name, m))
    return out


def _latest_look(runs: list[str]) -> str | None:
    """The most recently completed eval run (metrics.json is written last),
    ties broken by name: deterministic for a given eval folder."""
    def written(run: str) -> int:
        try:
            return (settings.eval_dir / run / "metrics.json").stat().st_mtime_ns
        except OSError:
            return -1

    return max(runs, key=lambda r: (written(r), r), default=None)


def _load_rgb(path: Path):
    from infer.pipeline import load_photo_rgb

    return load_photo_rgb(path)


def _verdict_key(v: dict) -> tuple[str, tuple[str, ...]]:
    return v["verdict"], tuple(v["classes"])


def _pick_boundary_photos(photos: list[dict], t: float, n: int) -> list[dict]:
    """Photos nearest the threshold, alternating N.iO side / i.O side."""
    above = sorted((p for p in photos if photo_score(p["scores"]) >= t),
                   key=lambda p: (photo_score(p["scores"]) - t, p["capture_id"]))
    below = sorted((p for p in photos if photo_score(p["scores"]) < t),
                   key=lambda p: (t - photo_score(p["scores"]), p["capture_id"]))
    out = []
    while (above or below) and len(out) < len(photos):
        for side in (above, below):
            if side:
                out.append(side.pop(0))
    return out


def check_parity(
    model_path: Path, adapter: str, threshold: float, class_names: list[str],
    reference_run: str | None, n_photos: int = 5,
) -> dict:
    """Deployed inference (floor = t) vs the verdicts the evaluation measured."""
    from eval.run_eval import load_predictions
    from infer.pipeline import TiledDetector

    if n_photos <= 0:
        logger.warning("parity check SKIPPED (verify_photos=0) — recorded in the card")
        return {"photos": 0, "verdict_mismatches": None, "mode": "skipped"}
    live = TiledDetector(model_path, adapter, conf_floor=threshold)
    checked, mismatches, boundary, score_mismatch = [], [], [], 0

    def compare(cid, expected_det, got_det, tol):
        nonlocal score_mismatch
        exp = photo_verdict(*expected_det, threshold, class_names)
        got = photo_verdict(got_det.boxes, got_det.scores, got_det.labels, threshold, class_names)
        checked.append(cid)
        exp_s = sorted((s for s in expected_det[1] if s >= threshold), reverse=True)
        got_s = sorted((s for s in got_det.scores if s >= threshold), reverse=True)
        if len(exp_s) != len(got_s) or any(abs(a - b) > tol for a, b in zip(exp_s, got_s)):
            score_mismatch += 1
        if _verdict_key(exp) == _verdict_key(got):
            return
        # which classes flip, and does each one sit within the stored-score
        # precision of t (on the reference side or the live side)?
        def top(scores, labels, k):
            return max((s for s, c in zip(scores, labels) if class_names[c] == k), default=0.0)

        flips = set(exp["classes"]) ^ set(got["classes"])
        near = tol > 0 and all(
            min(abs(top(expected_det[1], expected_det[2], k) - threshold),
                abs(top(got_det.scores, got_det.labels, k) - threshold)) <= tol
            for k in flips
        )
        item = {"capture_id": cid, "expected": list(_verdict_key(exp)), "live": list(_verdict_key(got))}
        (boundary if near else mismatches).append(item)

    if reference_run:
        doc = load_predictions(reference_run)
        if list(doc["class_names"]) != list(class_names):
            raise ExportError(f"parity reference {reference_run} has classes {doc['class_names']}")
        tol = 0.0 if doc.get("format_version", 1) >= 2 else V1_SCORE_TOLERANCE
        paths = _photo_paths([p["capture_id"] for p in doc["photos"]])
        for p in _pick_boundary_photos(doc["photos"], threshold, n_photos):
            if len(checked) >= n_photos:
                break
            path = paths.get(p["capture_id"])
            if path is None or not Path(path).is_file():
                continue
            compare(p["capture_id"], (p["boxes"], p["scores"], p["labels"]),
                    live.detect(_load_rgb(path)), tol)
        mode = "reference"
    else:
        full = TiledDetector(model_path, adapter, conf_floor=0.001)
        ids = _dev_photo_ids(n_photos)
        paths = _photo_paths(ids)
        for cid in ids:
            if len(checked) >= n_photos:
                break
            path = paths.get(cid)
            if path is None or not Path(path).is_file():
                continue
            rgb = _load_rgb(path)
            ref = full.detect(rgb)
            compare(cid, (ref.boxes, ref.scores, ref.labels), live.detect(rgb), 0.0)
        mode = "floor-equivalence (no eval of this model: floor t vs floor 0.001, live)"
    if not checked:
        raise ExportError(
            "parity check found no photo files to run on — the photos must be on this "
            "machine (or pass --verify-photos 0, recorded as skipped)"
        )
    result = {
        "photos": len(checked), "verdict_mismatches": len(mismatches), "mode": mode,
        "reference_eval_run": reference_run, "capture_ids": checked,
        "score_mismatches": score_mismatch,
    }
    if mismatches:
        result["mismatches"] = mismatches
    if boundary:
        result["rounding_boundary"] = boundary
    return result


def _atomic_copy(src: Path, dest: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
            shutil.copyfileobj(inp, out, 1 << 20)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------- run


def run(
    bundle_dir: Path,
    operating_point: Path,
    eval_run: str | None = None,
    out_dir: Path = settings.exports_dir,
    name: str = "detector",
    allow_fold_model: bool = False,
    verify_photos: int = 5,
    no_test_eval: bool = False,
    force: bool = False,
) -> Path:
    """Verify and export; returns the written card path (<out_dir>/<name>.json)."""
    from eval.run_eval import op_model_problems

    bundle_dir, op_path, out_dir = Path(bundle_dir), Path(operating_point), Path(out_dir)
    if not _NAME_RE.match(name):
        raise ExportError(f"--name must be a plain file stem, got {name!r}")
    info = registry.bundle_info(bundle_dir)
    if info is None:
        raise ExportError(f"{bundle_dir} has no meta.json — import the bundle first (cli.py import-run)")
    model = bundle_dir / "model.onnx"
    if not model.is_file():
        raise ExportError(f"{model} not found")
    op = json.loads(op_path.read_text(encoding="utf-8"))
    split = load_split()
    problems: list[str] = []
    overrides: list[str] = []

    # ---- 1. identity: meta vs operating point ----
    schema = info["schema"]
    try:
        classes = schema_classes(schema)
    except KeyError:
        raise ExportError(f"bundle schema {schema!r} is unknown") from None
    if op.get("schema") != schema:
        problems.append(f"operating point is for schema {op.get('schema')!r}, the bundle {schema!r}")
    if list(op.get("class_names") or []) != classes:
        problems.append(f"operating point classes {op.get('class_names')} != {classes}")
    if info["classes"] and list(info["classes"]) != classes:
        problems.append(f"bundle classes {info['classes']} != schema {schema!r} {classes}")
    if info["split_id"] != split["split_id"] or op.get("split_id") not in (None, split["split_id"]):
        problems.append(f"split mismatch: bundle {info['split_id']!r}, operating point "
                        f"{op.get('split_id')!r}, frozen split {split['split_id']!r}")
    if info["tiler_version"] not in (None, TILER_VERSION):
        problems.append(f"bundle trained on tiler {info['tiler_version']!r}, inference is {TILER_VERSION!r}")
    if info["adapter"] not in KNOWN_ADAPTERS:
        problems.append(f"adapter {info['adapter']!r} is not one of {list(KNOWN_ADAPTERS)}")
    t = op.get("threshold")
    if not isinstance(t, (int, float)) or isinstance(t, bool) or not 0 < t < 1:
        problems.append(f"operating point threshold {t!r} is not in (0, 1)")
    if "sources" in op:  # v2 operating point: the test eval's op <-> model identity check
        problems += op_model_problems(op, info)
    elif allow_fold_model:
        overrides.append("v1 operating point (no family/sources): family and config "
                         "equivalence unverified")
    else:
        problems.append("v1 operating point (no family/sources) predates the choose-op guards — "
                        "re-run `cli.py choose-op --family <F>`")
    # ---- 2. dev_all only ----
    if info["fold"] != "dev_all":
        if allow_fold_model:
            overrides.append(f"allow_fold_model: exporting a {info['fold']} model, not the dev_all retrain")
        else:
            problems.append(f"{info['run_name']} is a {info['fold']} model — export the dev_all "
                            "retrain, or pass --allow-fold-model (recorded in the card)")
    if problems:
        raise ExportError("export refused:\n  - " + "\n  - ".join(problems))
    t = float(t)
    model_sha = registry.file_sha256(model)
    op_sha = registry.file_sha256(op_path)

    # ---- 3. test evaluation of exactly this model at exactly this op ----
    runs = dict(_eval_runs_of_model(model_sha))
    if eval_run is not None and eval_run not in runs:
        raise ExportError(f"eval run {eval_run!r} is not an evaluation of this model (sha "
                          f"{model_sha[:12]}…)")
    explicit_test = eval_run is not None and runs[eval_run].get("bucket") == "test"
    # an explicit non-test eval_run is the parity reference only
    tests = [eval_run] if explicit_test else [r for r, m in runs.items() if m.get("bucket") == "test"]

    def op_used(m: dict) -> str | None:
        return (m.get("operating_point") or {}).get("sha256")

    # the run must have RECORDED this operating point's sha: a matching
    # threshold alone could come from another op file (or none at all)
    matching = [r for r in tests if op_used(runs[r]) == op_sha
                and runs[r].get("threshold") is not None and abs(runs[r]["threshold"] - t) <= 1e-12]
    test_run = _latest_look(matching)  # further looks (other floor, relabel): the latest counts
    others = [r for r in tests if r not in matching]
    test_level1 = None
    # This model has been looked at on test at another operating point: shipping
    # this one without its own (ledger-recorded) test look would pick the
    # threshold after seeing test, so --no-test-eval cannot waive it.
    if others and (explicit_test or not test_run):
        raise ExportError(
            "test eval " + ", ".join(
                f"{r} (t={runs[r].get('threshold')}, op sha {(op_used(runs[r]) or 'not recorded')[:12]})"
                for r in others)
            + f" used another operating point than {op_path.name} (t={t}, sha {op_sha[:12]}) — "
            "evaluate this operating point on test (a second look, recorded in the test ledger); "
            "--no-test-eval cannot waive a test look the model already had")
    if test_run:
        if len(matching) > 1:
            logger.info(f"test evals of this model at this operating point: {sorted(matching)} — "
                        f"the latest, {test_run}, is recorded")
        tm = runs[test_run]
        if tm.get("guard_overridden"):
            raise ExportError(f"test eval {test_run} was run with guards overridden")
        test_level1 = tm.get("level1_verdict")
    elif no_test_eval:
        overrides.append("no_test_eval: no test evaluation of this model")
    else:
        raise ExportError(
            f"no test evaluation of this model (sha {model_sha[:12]}…) at this operating point — "
            f"run `cli.py eval --model {model} --bucket test --operating-point {op_path}` first, "
            "or pass --no-test-eval (recorded as null)"
        )

    # ---- 4. parity ----
    if eval_run is not None:
        reference = eval_run
    elif test_run:
        reference = test_run
    else:
        own = f"{info['family']}_{info['fold']}"
        v2 = [r for r, m in runs.items() if m.get("source")]
        reference = own if own in runs else (v2[0] if v2 else (next(iter(runs), None)))
    parity = check_parity(model, info["adapter"], t, classes, reference, verify_photos)
    logger.info(f"parity: {json.dumps(parity, ensure_ascii=False)}")
    if parity["verdict_mismatches"]:
        raise ExportError(
            f"parity FAILED: {parity['verdict_mismatches']} of {parity['photos']} photos get "
            f"another verdict from the deployed pipeline than eval {reference} measured: "
            f"{parity.get('mismatches')}"
        )

    # ---- 5. write: staging dir -> atomic replace ----
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_onnx, dest_card = out_dir / f"{name}.onnx", card_path_for(out_dir / f"{name}.onnx")
    if dest_onnx.exists() or dest_card.exists():
        if not force:
            raise ExportError(f"{dest_onnx.name}/{dest_card.name} already exist in {out_dir} — "
                              "pass --force to archive them and export")
        archive = out_dir / "archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive.mkdir(parents=True, exist_ok=True)
        for p in (dest_onnx, dest_card):
            if p.exists():
                shutil.move(str(p), str(archive / p.name))
        logger.warning(f"--force: previous export archived to {archive}")
    op_block = {
        "file": op_path.name, "sha256": op_sha, "family": op.get("family"),
        "rule": op.get("rule"), "target_sensitivity": op.get("target_sensitivity"),
        "oof_level1": op.get("oof_level1"),
    }
    card = {
        "contract_version": 1, "kind": "zander-detector",
        "model_file": dest_onnx.name, "model_sha256": model_sha,
        "adapter": info["adapter"], "schema": schema, "class_names": classes,
        "threshold": t, "conf_floor": t,
        "decision_rule": DECISION_RULE,
        "tiler_version": TILER_VERSION, "tile_size": TILE_SIZE, "stride": STRIDE,
        "operating_point": op_block,
        "source": {
            "bundle_run": info["run_name"], "fold": info["fold"], "weights": info["weights"],
            "bundle_sha256": info["bundle_sha256"], "dataset_id": info["dataset_id"],
            "dataset_fingerprint": info["dataset_fingerprint"], "split_id": info["split_id"],
            "snapshot_id": info["snapshot_id"], "git_sha": info["git_sha"],
            "framework": info["framework"], "versions": info["versions"],
        },
        "validation": {
            "test_eval_run": test_run, "test_level1": test_level1, "parity": parity,
            "overrides": overrides,
        },
        "verifier": None,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exported_by": f"{settings.toolchain_name} {settings.toolchain_version} cli export",
    }
    staging = Path(tempfile.mkdtemp(prefix=".export-", dir=str(out_dir)))
    try:
        staged_onnx = staging / dest_onnx.name
        _atomic_copy(model, staged_onnx)
        if registry.file_sha256(staged_onnx) != model_sha:
            raise ExportError("copied ONNX does not match the bundle's sha256")
        write_detector_card(card_path_for(staged_onnx), card)  # validates against the copy
        os.replace(staged_onnx, dest_onnx)
        os.replace(card_path_for(staged_onnx), dest_card)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    loaded = load_detector_card(dest_onnx)  # read back exactly as the inspector will
    for o in overrides:
        logger.warning(f"export override recorded in the card: {o}")
    logger.info(
        f"exported {info['run_name']} -> {dest_onnx} + {dest_card.name} "
        f"(t={loaded.threshold}, {loaded.adapter}, {list(loaded.class_names)}); copy both to "
        "DATA_ROOT/models/ (MODEL_PATH=/data/models/detector.onnx) and restart the inspector"
    )
    return dest_card
