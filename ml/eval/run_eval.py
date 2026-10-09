"""
Class-aware photo-level evaluation of one ONNX checkpoint on one bucket.

    python cli.py eval --model ../ml_data/bundles/<run>/model.onnx --bucket fold0
    python cli.py choose-op --family yolo26n_def3          # <F>_fold0..2 -> frozen t*
    python cli.py eval --model <devall>/model.onnx --bucket test --operating-point op.json

Buckets (split_v2, dataprep/split.py):
  fold0|fold1|fold2  validation fold of the model trained without it
  probe_xprod        photos holding the board-A Brücke/Fahne probe boxes, minus
                     test photos (per-box hits only — their other defects may be
                     trained)
  train_always       board B + A2 (always trained): memorisation check only
  test               ONE-SHOT, final (dev_all) model only — see the test ledger
  future             captures after the split freeze (chronological test)

Honesty rules enforced here (docs/CONTRACTS.md §8.2), all BEFORE any photo is
read or any ONNX session is created:
  - the bucket must exist; schema, adapter and split_id must match the model's
    bundle meta.json (next to the .onnx) — not overridable;
  - own-fold guard: a fold bucket only scores the model that was validated on
    it (meta.fold == bucket); dev_all models never score folds; a model without
    meta.json cannot be verified; test is for dev_all models;
  - run policy (train/ultralytics_run.py): a bundle with weights `best`, a
    non-zero `extra.patience` or `epochs_completed < epochs` holds a checkpoint
    selected on (or stopped by) its validation view. These and the own-fold
    guard are overridable with `allow_foreign_fold=True` (--allow-foreign-fold)
    for diagnostics: the run is stamped `guard_overridden` and never recorded in
    results_v2.csv; choose-op refuses such models unconditionally;
  - test / probe_xprod / future REQUIRE --operating-point (a threshold frozen
    on out-of-fold dev predictions, a number in (0, 1]); the threshold is never
    re-derived there. On test the op must carry choose-op provenance and belong
    to the model (`op_model_problems`: family + config equivalence with its
    fold bundles — export's identity check, run before the one-shot look);
  - the TEST LEDGER (committed ml/reports/test_ledger.csv) records every look
    at test, keyed by (model, operating point, conf floor, snapshot id, GT
    sha). Re-running the same key is fine (deterministic); another key is a
    second look at the one-shot set and is refused unless
    `second_look="<reason>"`, which is recorded. Test evals in ml_data/eval
    (shared across checkouts) count as looks too; a look's eval dir is never
    overwritten (a further look gets `<bundle_run>__test__look<k>`).
  - fold runs without an operating point report a self-tuned threshold,
    flagged as diagnostic; `choose-op` pools the three fold runs' predictions
    into the real dev operating point (distinct own-fold models required).

Writes ml_data/eval/<run_name>/{metrics.json, predictions.json (v2), report.md,
gallery.html} and appends one row to the committed ml/reports/results_v2.csv
(never for smoke or guard-overridden runs). Default run names follow
CONTRACTS §8.2: `<family>_fold<k>` on the model's own fold, else
`<bundle_run>__<bucket>` (`__look<k>` appended for a further test look). Model-vs-model decisions use ONLY these photo-level
numbers — never per-tile or in-training validation metrics.

predictions.json v2 = v1 + format_version, source{…}, per-photo width/height
(and det_scores for cascade runs). Scores are stored at full float precision,
so a threshold frozen from them is exactly reproducible by the deployed
detector (5-decimal rounding put the threshold-defining photo on a coin flip).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import registry
from config.settings import settings
from dataprep.ingest import load_manifest, photo_path, usable
from dataprep.remap import SHORT, project, schema_classes
from dataprep.split import bucket_of, load_split
from eval.gallery import write_gallery
from eval.photo_scorer import match_indices, score_all, threshold_for_sensitivity
from eval.scorer import FPPI_OPERATING, IOU_OPERATING, iou
from infer.pipeline import TiledDetector

logger = logging.getLogger("ml.eval")

FROZEN_OP_BUCKETS = {"test", "probe_xprod", "future"}
# Static list for the CLI; valid_buckets() derives the fold names from the split.
BUCKETS = ("fold0", "fold1", "fold2", "train_always", "test", "probe_xprod", "future")
NOT_HONEST_BUCKETS = {
    "probe_xprod": "only the per-box probe section is honest here — these photos' OTHER "
                   "defects may be in the model's training data; levels 1-3 are context only",
    "train_always": "board B is always trained: every number below is memorisation, not validation",
}
PREDICTIONS_FORMAT_VERSION = 2
RESULTS_FILE = "results_v2.csv"
RESULTS_FIELDS = [
    "date", "run_name", "model", "adapter", "schema", "bucket", "threshold",
    "threshold_source", "n_photos", "n_positive", "n_negative_seated",
    "photo_auroc", "escape", "escape_ci_lo", "escape_ci_hi",
    "fcr_seated", "fcr_seated_ci_lo", "fcr_seated_ci_hi", "fcr_handheld",
    "recall_nal_0.5fppi", "recall_nal_ci_lo", "recall_nal_ci_hi",
    "recall_bruecke_0.5fppi", "recall_fahne_0.5fppi", "recall_any_0.5fppi",
    "exact_set_acc", "ap50_nal", "probe_found", "mean_ms_per_photo",
    "tiler_version", "split_id", "snapshot_id", "dataset_fingerprint", "host",
]
# results_v2.csv holds honest evals only (README): train_always is memorisation;
# on probe_xprod only the per-box probe counts are honest (probe boxes are held
# out of every training view, the photos' other defects may be trained), so its
# photo/defect-level columns stay empty.
NOT_RECORDED_BUCKETS = {"train_always"}
PROBE_ONLY_BUCKETS = {"probe_xprod"}
_NOT_HONEST_ON_PROBE = (
    "n_positive", "n_negative_seated", "photo_auroc", "escape", "escape_ci_lo", "escape_ci_hi",
    "fcr_seated", "fcr_seated_ci_lo", "fcr_seated_ci_hi", "fcr_handheld",
    "recall_nal_0.5fppi", "recall_nal_ci_lo", "recall_nal_ci_hi", "recall_bruecke_0.5fppi",
    "recall_fahne_0.5fppi", "recall_any_0.5fppi", "exact_set_acc", "ap50_nal",
)
TEST_LEDGER_FIELDS = [
    "date", "run_name", "model_sha256", "operating_point_sha256", "bundle_run",
    "family", "schema", "threshold", "look", "reason",
    "conf_floor", "snapshot_id", "gt_sha256", "guard_overridden",
]
# choose-op / export / compare: fold models of one family must share these (CONTRACTS §8.3).
EQUIV_KEYS = ("arch", "dataset_id", "epochs", "imgsz", "batch", "augment", "final_weights", "extra")
# `extra` keys (model.train() kwargs) that do not change the trained weights —
# validation, logging and runtime only. dev_all configs differ from the fold
# configs by extra.val and must stay equivalent. Seeds are never compared.
NON_TRAINING_EXTRA = frozenset({
    "val", "val_fraction", "plots", "verbose", "workers", "cache", "device", "exist_ok",
    "save_period",
})


class EvalGuardError(RuntimeError):
    """An evaluation (or operating point) that would be dishonest or inconsistent."""


def valid_buckets(split: dict | None = None) -> tuple[str, ...]:
    split = split or load_split()
    folds = [f"fold{i}" for i in sorted(split["folds"], key=int)]
    return (*folds, "train_always", "test", "probe_xprod", "future")


def bucket_photos(bucket: str, schema: str) -> tuple[list[dict], list[dict]]:
    """(manifest entries of the bucket, probe boxes relevant to it).

    probe_xprod = the photos holding probe boxes MINUS test photos: the one-shot
    test set is only ever looked at through `--bucket test`.
    """
    manifest, split = load_manifest(), load_split()
    if bucket not in valid_buckets(split):
        raise ValueError(f"unknown bucket {bucket!r} — one of {list(valid_buckets(split))}")
    classes = schema_classes(schema)
    probe = [p for p in split["probe_boxes"] if p["class"] in classes]
    probe_ids = {p["capture_id"] for p in probe}
    entries = []
    for e in manifest["entries"]:
        if not usable(e):
            continue
        b = bucket_of(e, split)
        if bucket == "probe_xprod":
            if e["capture_id"] in probe_ids and b != "test":
                entries.append(e)
        elif b == bucket:
            entries.append(e)
    if not entries:
        raise ValueError(f"no usable photos in bucket {bucket!r} (schema {schema})")
    ids = {e["capture_id"] for e in entries}
    return entries, [p for p in probe if p["capture_id"] in ids]


def _gt_record(e: dict, schema: str, split: dict) -> dict:
    train, other = project(e["boxes"], schema)
    return {
        "capture_id": e["capture_id"],
        "group": split["assignment"].get(e["capture_id"], {}).get("group", e["capture_id"]),
        "board": e["board"],
        "pose": e["pose"],
        "gt_verdict": e["verdict"],
        "width": e.get("width"),
        "height": e.get("height"),
        "gt": [(c, tuple(b)) for c, b in train],
        "other_gt": [(n, tuple(b)) for n, b in other],
    }


# Public alias (CONTRACTS §8.2 freezes `_gt_record`; this is the same function).
gt_record = _gt_record


def valid_threshold(t) -> bool:
    """A frozen threshold: a finite real number in (0, 1] (never null/bool/NaN —
    a null threshold would make score_all self-tune on the bucket)."""
    return (isinstance(t, (int, float)) and not isinstance(t, bool)
            and math.isfinite(t) and 0 < t <= 1)


# choose-op provenance a test-bucket operating point must carry (CONTRACTS §8.3)
OP_PROVENANCE_KEYS = ("family", "sources", "target_sensitivity")


def _load_op(path: Path, schema: str, *, bucket: str | None = None) -> dict:
    op = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(op, dict):
        raise ValueError("not a JSON object")
    if op.get("schema") != schema:
        raise ValueError(f"operating point is for schema {op.get('schema')!r}, not {schema!r}")
    if op.get("class_names") is not None and list(op["class_names"]) != schema_classes(schema):
        raise ValueError(f"operating point classes {op['class_names']} != schema {schema!r}")
    split_id = load_split()["split_id"]
    if op.get("split_id") is not None and op["split_id"] != split_id:
        raise ValueError(f"operating point was frozen on {op['split_id']!r}, not {split_id!r}")
    if not valid_threshold(op.get("threshold")):
        raise ValueError(f"threshold {op.get('threshold')!r} is not a number in (0, 1] — a "
                         "frozen operating point needs one (`cli.py choose-op`)")
    if bucket == "test":
        missing = [k for k in OP_PROVENANCE_KEYS if not op.get(k)]
        if missing:
            raise ValueError(
                f"no choose-op provenance ({', '.join(missing)} missing): the one-shot test set "
                "only takes an operating point frozen by `cli.py choose-op --family <F>` — "
                "re-run choose-op"
            )
    return op


def _bundle_meta(model_path: Path) -> dict:
    meta = model_path.parent / "meta.json"
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}


# ------------------------------------------------------------------ guards


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def training_problems(info: dict) -> list[str]:
    """Run-policy violations (train/ultralytics_run.py: last epoch, patience 0)
    that make a model's out-of-fold numbers dishonest: a checkpoint selected on
    its validation view (weights best, early stopping) or a run that stopped
    before its configured epochs. DEIM bundles (no extra.patience, always last)
    pass."""
    name = info["run_name"]
    out = []
    if info.get("weights") == "best":
        out.append(f"{name} bundles weights 'best' — a checkpoint selected on its validation "
                   "view (selection leak; run policy: the last epoch)")
    extra = (info.get("config") or {}).get("extra")
    patience = extra.get("patience") if isinstance(extra, dict) else None
    if patience not in (None, 0):
        out.append(f"{name} trained with extra.patience={patience!r} — early stopping on its "
                   "validation view (selection leak; run policy: patience 0)")
    done, epochs = (info.get("meta") or {}).get("epochs_completed"), info.get("epochs")
    if _is_int(done) and _is_int(epochs) and done < epochs:
        out.append(f"{name} stopped after {done} of {epochs} epochs (early stop or truncated run)")
    return out


def check_eval_guards(
    model_path: Path, adapter_name: str, schema: str, bucket: str,
    *, allow_foreign_fold: bool = False,
) -> tuple[dict | None, list[str]]:
    """Refuse dishonest or inconsistent evaluations (see module doc).

    Returns (normalised bundle info or None, overridden honesty guards)."""
    split = load_split()
    if bucket not in valid_buckets(split):
        raise EvalGuardError(f"unknown bucket {bucket!r} — one of {list(valid_buckets(split))}")
    info = registry.bundle_info(Path(model_path).parent)
    problems: list[str] = []  # correctness: never overridable
    honesty: list[str] = []   # overridable with allow_foreign_fold (stamped)
    try:
        classes = schema_classes(schema)
    except KeyError:
        classes = None
        problems.append(f"unknown schema {schema!r} (bundle meta.json has none?) — one of "
                        "['def3', 'nal1']")
    if info is None:
        honesty.append(
            f"no meta.json next to {Path(model_path).name}: the model's fold, schema "
            "and split cannot be verified"
        )
    else:
        name = info["run_name"]
        if info["schema"] != schema:
            problems.append(f"model {name} is schema {info['schema']!r}, not {schema!r}")
        elif classes is not None and info["classes"] and list(info["classes"]) != classes:
            problems.append(f"model {name} classes {info['classes']} != schema {schema!r} {classes}")
        if info["adapter"] and info["adapter"] != adapter_name:
            problems.append(f"model {name} needs adapter {info['adapter']!r}, not {adapter_name!r}")
        if info["split_id"] != split["split_id"]:
            problems.append(
                f"model {name} was trained on split {info['split_id']!r}, but the frozen "
                f"split is {split['split_id']!r} — its folds/test are not this benchmark's"
            )
        fold = info["fold"]
        if bucket.startswith("fold"):
            if fold == "dev_all":
                honesty.append(f"{name} is a dev_all model: {bucket} is its training data")
            elif fold != bucket:
                honesty.append(
                    f"{name} validates on {fold}, not {bucket} ({bucket} is in its training data)"
                )
        elif bucket == "test" and fold != "dev_all":
            honesty.append(
                f"test is the one-shot evaluation of the FINAL dev_all model; {name} is a {fold} model"
            )
        honesty += training_problems(info)
    binding = registry.model_file_problem(Path(model_path).parent, Path(model_path))
    if binding:  # meta.json would describe another model than the one scored
        problems.append(binding)
    if problems:
        raise EvalGuardError("refused: " + "; ".join(problems))
    if honesty:
        if not allow_foreign_fold:
            raise EvalGuardError(
                "refused (not an honest evaluation): " + "; ".join(honesty)
                + " — pass --allow-foreign-fold to evaluate anyway (stamped guard_overridden, "
                "never recorded in results_v2.csv)"
            )
        logger.warning("GUARD OVERRIDDEN: " + "; ".join(honesty))
    return info, honesty


def test_ledger_path() -> Path:
    return settings.test_ledger_file


# A test look is identified by the model, the operating point, the confidence
# floor and the ground truth it was scored against: changing any of them is a
# new test number, i.e. another look at the one-shot set.
LEDGER_KEY_FIELDS = ("model_sha256", "operating_point_sha256", "conf_floor", "snapshot_id",
                     "gt_sha256")


def _key_value(field: str, value) -> str:
    if value is None or value == "":
        return ""
    if field == "conf_floor":
        try:
            return repr(float(value))
        except (TypeError, ValueError):
            return str(value)
    return str(value).strip().lower() if field.endswith("sha256") else str(value).strip()


def look_key(look: dict) -> tuple[str, ...]:
    """Normalised ledger key of a ledger row / eval-dir look / key dict."""
    return tuple(_key_value(f, look.get(f)) for f in LEDGER_KEY_FIELDS)


def ground_truth_sha256(records: list[dict]) -> str:
    """Identity of the ground truth an eval scored (the bucket's _gt_record list):
    a relabel, a review fix or a quarantine change gives another sha."""
    recs = sorted(records, key=lambda r: r["capture_id"])
    blob = json.dumps(recs, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cascade_identity(model_sha: str, stacker_sha: str) -> str:
    """CONTRACTS §8.4: the ledger's model key of a detector+verifier cascade."""
    return hashlib.sha256(f"cascade:{model_sha}:{stacker_sha}".encode()).hexdigest()


def _eval_dir_look(d: Path) -> dict | None:
    """The test look an eval dir holds (its metrics.json, bucket test), else None."""
    mpath = Path(d) / "metrics.json"
    if not mpath.is_file():
        return None
    try:
        m = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning(f"{mpath}: unreadable ({e}) — not counted as a test look")
        return None
    if not isinstance(m, dict) or m.get("bucket") != "test":
        return None
    src = m.get("source") or {}
    model_sha = src.get("model_sha256")
    if src.get("kind") == "cascade" and model_sha and src.get("stacker_sha256"):
        model_sha = _cascade_identity(model_sha, src["stacker_sha256"])
    return {"run_name": Path(d).name, "model_sha256": model_sha,
            "operating_point_sha256": (m.get("operating_point") or {}).get("sha256"),
            "conf_floor": src.get("conf_floor"), "snapshot_id": m.get("snapshot_id"),
            "gt_sha256": m.get("gt_sha256")}


def eval_dir_test_looks() -> list[dict]:
    """Every test evaluation in ml_data/eval. That folder is shared by all
    checkouts and worktrees while the ledger is per checkout (and `git checkout
    -- reports/` drops its rows), so these count as recorded looks too."""
    if not settings.eval_dir.is_dir():
        return []
    return [look for d in sorted(settings.eval_dir.iterdir()) if d.is_dir()
            for look in [_eval_dir_look(d)] if look]


def _read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("model_sha256")]


def check_test_ledger(model_sha: str, op_sha: str, second_look: str | None = None, *,
                      conf_floor: float | None = None, snapshot_id: str | None = None,
                      gt_sha256: str | None = None) -> dict:
    """One-shot rule for the test bucket (ml/reports/test_ledger.csv + the test
    evals in ml_data/eval).

    Key = (model sha, operating point sha, conf floor, snapshot id, GT sha).
    First look, or a rerun of a recorded key -> allowed. Any other key ->
    refused unless `second_look` gives a reason (then recorded). A rerun found
    only in ml_data/eval is re-recorded in the ledger. Returns {"look":
    first|rerun|second_look, "reason": str, "record": bool}."""
    path = test_ledger_path()
    rows = _read_ledger(path)
    key = look_key({"model_sha256": model_sha, "operating_point_sha256": op_sha,
                    "conf_floor": conf_floor, "snapshot_id": snapshot_id, "gt_sha256": gt_sha256})
    in_ledger = {look_key(r) for r in rows}
    unrecorded = [lk for lk in eval_dir_test_looks() if look_key(lk) not in in_ledger]
    if not rows and not unrecorded:
        return {"look": "first", "reason": "", "record": True}
    if key in in_ledger:
        return {"look": "rerun", "reason": "", "record": False}
    same = [lk for lk in unrecorded if look_key(lk) == key]
    if same:
        reason = (f"restored: look found in ml_data/eval/{same[0]['run_name']} but missing "
                  f"from {path.name}")
        logger.warning(f"rerun of an unrecorded test look — {reason}")
        return {"look": "rerun", "reason": reason, "record": True}
    reason = (second_look or "").strip()
    if reason:
        logger.warning(f"SECOND LOOK at the test set (recorded in {path.name}): {reason}")
        return {"look": "second_look", "reason": reason, "record": True}
    looks = rows + unrecorded
    last = looks[-1]
    extra = (f"; {len(unrecorded)} of them only in ml_data/eval, missing from {path.name}"
             if unrecorded else "")
    raise EvalGuardError(
        f"the one-shot test set was already evaluated ({len(looks)} recorded look(s){extra}; "
        f"last {last['run_name']}: model {(last.get('model_sha256') or '?')[:12]}…, operating "
        f"point {(last.get('operating_point_sha256') or '?')[:12]}…). Another model, operating "
        f"point, --conf or ground truth is a second look — if that is deliberate pass "
        f"--second-look \"<reason>\" (recorded in {path})"
    )


def _ledger_fields(path: Path) -> list[str]:
    """The ledger's columns, migrating an older header in place first (rows kept,
    new columns empty). New file -> TEST_LEDGER_FIELDS."""
    if not path.exists() or path.stat().st_size == 0:
        return list(TEST_LEDGER_FIELDS)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    fields = list(TEST_LEDGER_FIELDS) + [c for c in header if c not in TEST_LEDGER_FIELDS]
    if header == fields:
        return fields
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=fields, restval="", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    _atomic_write_text(path, out.getvalue())
    logger.warning(f"{path.name}: header migrated to {len(fields)} columns ({len(rows)} row(s) kept)")
    return fields


def record_test_look(look: dict, *, run_name: str, model_sha: str, op_sha: str,
                     bundle_run: str | None, family: str | None, schema: str,
                     threshold: float, conf_floor: float | None = None,
                     snapshot_id: str | None = None, gt_sha256: str | None = None,
                     guard_overridden: list[str] | None = None) -> None:
    if look["look"] == "rerun" and not look.get("record"):
        return
    path = test_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _ledger_fields(path)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a+b") as fb:  # a missing final newline would glue two rows
        if not new:
            fb.seek(-1, os.SEEK_END)
            if fb.read(1) != b"\n":
                fb.write(b"\n")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow({
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_name": run_name, "model_sha256": model_sha,
            "operating_point_sha256": op_sha, "bundle_run": bundle_run or "",
            "family": family or "", "schema": schema, "threshold": threshold,
            "look": look["look"], "reason": look["reason"],
            "conf_floor": "" if conf_floor is None else conf_floor,
            "snapshot_id": snapshot_id or "", "gt_sha256": gt_sha256 or "",
            "guard_overridden": "; ".join(guard_overridden or []),
        })
        f.flush()
        os.fsync(f.fileno())  # the audit trail of a one-shot look must survive a crash
    logger.warning(f"test look recorded in {path} ({look['look']}) — commit it")


def _claim_run_name(explicit: str | None, default: str, bucket: str, key: dict) -> str:
    """An eval dir holding a test look is never overwritten by another look (or
    by a non-test eval): a further test look gets `<default>__look<k>`; an
    explicit --run-name that would overwrite one is refused."""
    want = look_key(key) if bucket == "test" else None
    name, k = explicit or default, 1
    while True:
        held = _eval_dir_look(settings.eval_dir / name)
        if held is None or (want is not None and look_key(held) == want):
            return name
        if explicit:
            raise EvalGuardError(
                f"ml_data/eval/{name} holds another test look (model "
                f"{(held.get('model_sha256') or '?')[:12]}…) — a look is never overwritten: "
                "choose another --run-name"
            )
        k += 1
        name = f"{default}__look{k}"


def default_run_name(info: dict | None, model_path: Path, schema: str, bucket: str) -> str:
    """CONTRACTS §8.2: `<family>_fold<k>` on the model's own fold, else
    `<bundle_run>__<bucket>`; models without meta keep the old timestamped name."""
    if info:
        if bucket == info["fold"]:
            return f"{info['family']}_{bucket}"
        return f"{info['run_name']}__{bucket}"
    return f"{Path(model_path).stem}_{schema}_{bucket}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"


# ---------------------------------------------------------------- evaluate


def run(
    model_path: Path,
    adapter_name: str,
    schema: str,
    bucket: str,
    operating_point: Path | None = None,
    run_name: str | None = None,
    conf_floor: float = 0.001,
    *,
    allow_foreign_fold: bool = False,
    second_look: str | None = None,
    record: bool = True,
    gallery: bool = True,
) -> dict:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise EvalGuardError(f"model not found: {model_path}")
    # ---- guards: nothing below reads a photo or opens an ONNX session first ----
    info, overridden = check_eval_guards(
        model_path, adapter_name, schema, bucket, allow_foreign_fold=allow_foreign_fold
    )
    classes = schema_classes(schema)
    try:
        op = _load_op(operating_point, schema, bucket=bucket) if operating_point else None
    except (ValueError, KeyError, TypeError, OSError) as e:
        raise EvalGuardError(f"operating point {operating_point}: {e}") from None
    if second_look and bucket != "test":
        logger.warning("--second-look only applies to the test bucket — ignored")
    if bucket in FROZEN_OP_BUCKETS and op is None:
        raise EvalGuardError(
            f"bucket {bucket!r} needs --operating-point: its threshold must be frozen "
            "on out-of-fold dev predictions (`python cli.py choose-op`), never tuned here"
        )
    if op is not None and conf_floor > op["threshold"]:
        raise EvalGuardError(
            f"--conf {conf_floor} is above the frozen threshold {op['threshold']}: detections "
            "between them would be cut off and change the verdicts"
        )
    if bucket == "test" and info is not None:
        # the same identity check export runs — but BEFORE the one-shot look is spent
        mismatch = op_model_problems(op, info)
        if mismatch:
            raise EvalGuardError(
                f"refused: operating point {Path(operating_point).name} does not belong to "
                f"{info['run_name']}:\n  - " + "\n  - ".join(mismatch)
            )
    model_sha = registry.file_sha256(model_path)
    op_sha = registry.file_sha256(operating_point) if operating_point else None
    # the bucket's ground truth (manifest + split only — still no photo read): part of the look key
    manifest, split = load_manifest(), load_split()
    entries, probe = bucket_photos(bucket, schema)
    gt = [_gt_record(e, schema, split) for e in entries]
    gt_sha = ground_truth_sha256(gt)
    key = {"model_sha256": model_sha, "operating_point_sha256": op_sha, "conf_floor": conf_floor,
           "snapshot_id": manifest["snapshot_id"], "gt_sha256": gt_sha}
    look = None
    if bucket == "test":
        look = check_test_ledger(model_sha, op_sha, second_look, conf_floor=conf_floor,
                                 snapshot_id=manifest["snapshot_id"], gt_sha256=gt_sha)
        logger.warning(
            "TEST bucket: the one-shot honest number for the FINAL model. Do not use "
            "it for model selection or threshold tuning."
        )
    if bucket == "train_always":
        logger.warning("train_always photos are always trained on: memorisation check only")

    run_name = _claim_run_name(run_name, default_run_name(info, model_path, schema, bucket),
                               bucket, key)
    out_dir = settings.eval_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = TiledDetector(model_path, adapter_name, conf_floor=conf_floor)
    n_model = getattr(detector.adapter, "num_classes", None)
    if n_model is not None and n_model != len(classes):
        raise ValueError(
            f"model outputs {n_model} classes but schema {schema!r} has {len(classes)}"
        )

    photos, total_ms = [], 0.0
    for n, (e, gt_rec) in enumerate(zip(entries, gt), 1):
        det = detector.detect_file(photo_path(e))
        bad = [c for c in det.labels if not 0 <= c < len(classes)]
        if bad:  # end-to-end exports have no static class count to check up front
            raise ValueError(
                f"model predicted class index {max(bad, key=abs)} on {e['capture_id']}, but "
                f"schema {schema!r} has {len(classes)} classes — wrong --schema or model"
            )
        total_ms += det.elapsed_ms
        photos.append({**gt_rec, "boxes": det.boxes, "scores": det.scores, "labels": det.labels})
        if n % 10 == 0 or n == len(entries):
            logger.info(f"evaluated {n}/{len(entries)} photos")

    metrics = score_all(
        photos, classes,
        threshold=op["threshold"] if op else None,
        threshold_source=f"frozen: {operating_point}" if op else "",
        probe=probe,
    )
    source = {
        "kind": "detector",
        "bundle_run": info["run_name"] if info else None,
        "family": info["family"] if info else None,
        "fold": info["fold"] if info else None,
        "model_sha256": model_sha,
        "model_resolved": str(model_path.resolve()),
        "adapter": adapter_name,
        "conf_floor": conf_floor,
        "verifier_run": None,
        "stacker_sha256": None,
    }
    metrics.update({
        "run_name": run_name, "model": str(model_path), "adapter": adapter_name,
        "schema": schema, "bucket": bucket, "n_photos": len(photos),
        "mean_ms_per_photo": round(total_ms / len(photos), 1),
        "tiler_version": detector.tiler_version, "split_id": split["split_id"],
        "snapshot_id": manifest["snapshot_id"], "gt_sha256": gt_sha,
        "dataset_fingerprint": (info or {}).get("dataset_fingerprint")
        or _bundle_meta(model_path).get("dataset_fingerprint"),
        "host": platform.node(),
        "source": source,
    })
    if op:
        metrics["operating_point"] = {
            "file": str(operating_point), "sha256": op_sha, "family": op.get("family"),
            "threshold": op["threshold"],
        }
    if overridden:
        metrics["guard_overridden"] = overridden
    if look is not None:
        metrics["test_look"] = {"look": look["look"], "reason": look["reason"],
                                "ledger": test_ledger_path().name}
        record_test_look(
            look, run_name=run_name, model_sha=model_sha, op_sha=op_sha,
            bundle_run=source["bundle_run"], family=source["family"], schema=schema,
            threshold=op["threshold"], conf_floor=conf_floor,
            snapshot_id=manifest["snapshot_id"], gt_sha256=gt_sha, guard_overridden=overridden,
        )
    smoke = registry.is_smoke(run_name) or registry.is_smoke(source["bundle_run"])
    if record and (smoke or overridden):
        logger.info(
            "not recorded in results_v2.csv: "
            + ("smoke run" if smoke else "guard overridden")
        )
    _write_outputs(out_dir, metrics, photos, record=record and not smoke and not overridden,
                   gallery=gallery, source=source)
    return metrics


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _minimal_source(metrics: dict) -> dict:
    """source{} for callers that pass none (pre-v2 call sites)."""
    return {
        "kind": "detector", "bundle_run": None, "family": None, "fold": None,
        "model_sha256": None, "model_resolved": str(metrics.get("model")),
        "adapter": metrics.get("adapter"), "conf_floor": None,
        "verifier_run": None, "stacker_sha256": None,
    }


def _photo_record(p: dict) -> dict:
    rec = {k: p[k] for k in ("capture_id", "group", "board", "pose", "gt_verdict")}
    if p.get("width") is not None:
        rec["width"], rec["height"] = p["width"], p["height"]
    rec.update({
        "gt": [[c, [round(v, 1) for v in b]] for c, b in p["gt"]],
        "other_gt": [[n, [round(v, 1) for v in b]] for n, b in p["other_gt"]],
        "boxes": [[round(v, 1) for v in b] for b in p["boxes"]],
        # full precision: t* is one of these values and must stay reproducible
        "scores": [float(s) for s in p["scores"]],
        "labels": [int(c) for c in p["labels"]],
    })
    if p.get("det_scores") is not None:
        rec["det_scores"] = [float(s) for s in p["det_scores"]]
    return rec


def _write_outputs(
    out_dir: Path, metrics: dict, photos: list[dict],
    *, record: bool = True, gallery: bool = True, source: dict | None = None,
) -> None:
    """metrics.json + predictions.json (v2) + report.md (+ gallery.html, + a
    results_v2.csv row). `source` defaults to metrics["source"]. metrics.json is
    written LAST: it marks a complete run (eval-matrix skips runs that have it)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = source if source is not None else metrics.get("source")
    doc = {
        "format_version": PREDICTIONS_FORMAT_VERSION,
        "schema": metrics["schema"], "class_names": metrics["class_names"],
        "bucket": metrics["bucket"], "model": metrics["model"],
        "source": source or _minimal_source(metrics),
    }
    if metrics.get("guard_overridden"):
        doc["guard_overridden"] = metrics["guard_overridden"]
    doc["photos"] = [_photo_record(p) for p in photos]
    _atomic_write_text(out_dir / "predictions.json", json.dumps(doc, ensure_ascii=False))
    report = format_report(metrics)
    _atomic_write_text(out_dir / "report.md", report)
    if gallery:
        _gallery(out_dir, metrics, photos)
    if record:
        _append_results_row(metrics)
    _atomic_write_text(out_dir / "metrics.json",
                       json.dumps(metrics, indent=1, ensure_ascii=False))
    logger.info("\n" + report)
    logger.info(f"artifacts: {out_dir}")


# Public alias (CONTRACTS §8.2 freezes `_write_outputs`; this is the same function).
write_outputs = _write_outputs


def load_predictions(run_name: str) -> dict:
    """predictions.json of an eval run, v1 or v2 (v1 gets format_version 1 and
    source None; per-photo width/height may be absent)."""
    path = settings.eval_dir / run_name / "predictions.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, list):
        raise ValueError(f"{path} is a pre-v2 (split_v1) predictions list — re-evaluate the model")
    doc.setdefault("format_version", 1)
    doc.setdefault("source", None)
    for p in doc["photos"]:
        p["gt"] = [(c, tuple(b)) for c, b in p["gt"]]
        p["other_gt"] = [(n, tuple(b)) for n, b in p["other_gt"]]
        p["boxes"] = [tuple(b) for b in p["boxes"]]
    return doc


def run_source(doc: dict) -> dict:
    """Provenance of a loaded predictions doc: the v2 `source`, completed (for
    v1 runs: reconstructed) from the recorded model file and its bundle meta."""
    src = dict(doc.get("source") or {})
    model = registry.resolve_model_path(src.get("model_resolved") or doc.get("model"))
    if model is not None:
        src["model_resolved"] = str(model)
        if not src.get("model_sha256"):
            src["model_sha256"] = registry.file_sha256(model)
    info = source_bundle_info(src)
    if info:
        for key, value in (("bundle_run", info["run_name"]), ("family", info["family"]),
                           ("fold", info["fold"]), ("adapter", info["adapter"])):
            if not src.get(key):
                src[key] = value
    src.setdefault("kind", "detector")
    return src


def source_bundle_info(src: dict) -> dict | None:
    """Bundle meta of an eval source: next to the resolved model, else the
    imported bundle dir ml_data/bundles/<bundle_run>/."""
    dirs = []
    if src.get("model_resolved"):
        dirs.append(Path(src["model_resolved"]).parent)
    if src.get("bundle_run"):
        dirs.append(settings.bundles_dir / src["bundle_run"])
    for d in dirs:
        if (d / "meta.json").is_file():
            return registry.bundle_info(d)
    return None


def stale_eval_problem(run: str, src: dict, info: dict | None) -> str | None:
    """choose-op / compare / summary judge an eval run by its bundle AS IT IS
    NOW: if the bundle no longer holds the model the eval scored (re-import
    --force, a hand-copied model), its meta/config describe another model."""
    recorded = str(src.get("model_sha256") or "").lower()
    if info is None or not recorded:
        return None
    known = registry.bundle_model_shas(info)
    if known and recorded not in known:
        return (f"{run}: stale eval — it scored model {recorded[:12]}…, which bundle "
                f"{info['run_name']} no longer holds (now "
                f"{', '.join(s[:12] + '…' for s in sorted(known))}) — re-run eval")
    return None


def source_problems(run: str, src: dict, info: dict | None) -> list[str]:
    """Why one fold eval run cannot enter a pooled out-of-fold set (choose-op,
    compare, summary): a stale eval, a model.onnx that is not the one its
    meta.json describes, or a run-policy violation. `info` = source_bundle_info(src)."""
    if info is None:
        return []
    out = []
    stale = stale_eval_problem(run, src, info)
    if stale:
        out.append(stale)
    binding = registry.model_file_problem(info["bundle_dir"])
    if binding:
        out.append(f"{run}: {binding}")
    return out + [f"{run}: {p}" for p in training_problems(info)]


def check_op_sources(op: dict, op_name: str, docs: list[dict]) -> None:
    """A frozen operating point applied to fold runs (summary, compare) must have
    been frozen from exactly those runs' models (`sources[].model_sha256` per
    fold) — else it is stale. Smoke operating points only warn; v1 files (no
    sources) cannot be checked. dev_all models are not sources: their match is
    op_model_problems (test eval, export)."""
    if "sources" not in op:
        logger.warning(f"{op_name}: v1 operating point (no sources) — cannot verify it was "
                       "frozen from these models")
        return
    by_fold = {s.get("fold"): str(s.get("model_sha256") or "").lower()
               for s in op.get("sources") or [] if isinstance(s, dict)}
    bad = []
    for d in docs:
        src = run_source(d)
        fold = str(src.get("fold") or "")
        if not fold.startswith("fold"):
            continue
        want, got = by_fold.get(fold), str(src.get("model_sha256") or "").lower()
        if not want:
            bad.append(f"{fold}: the operating point has no source for it")
        elif got != want:
            bad.append(f"{fold}: the run scored model {got[:12] or '?'}…, the operating point "
                       f"was frozen from {want[:12]}…")
    if not bad:
        return
    msg = (f"{op_name} was frozen from other models than these runs ({'; '.join(bad)}) — "
           "stale operating point: re-run `cli.py choose-op --force`")
    if op.get("allow_smoke") or registry.is_smoke(op.get("family")) \
            or any(registry.is_smoke(r) for r in op.get("runs") or []):
        logger.warning(msg + " (smoke operating point: warning only)")
        return
    raise EvalGuardError(msg)


# ----------------------------------------------------------- operating point


def config_signature(info: dict) -> dict:
    """The recipe keys fold models of one family (and its dev_all retrain) must share."""
    cfg, meta = info.get("config") or {}, info.get("meta") or {}
    extra = cfg.get("extra") if isinstance(cfg.get("extra"), dict) else {}
    return {
        "arch": cfg.get("model") or meta.get("model"),
        "dataset_id": info.get("dataset_id"),
        "epochs": cfg.get("epochs", meta.get("epochs")),
        "imgsz": cfg.get("imgsz"),
        # the batch the trainer really used (Ultralytics' OOM auto-batch may change it)
        "batch": meta["batch_effective"] if meta.get("batch_effective") is not None
        else cfg.get("batch"),
        "augment": meta.get("augmentation_online") or cfg.get("augment"),
        "final_weights": cfg.get("final_weights") or info.get("weights"),
        "extra": {k: v for k, v in sorted(extra.items()) if k not in NON_TRAINING_EXTRA},
    }


def op_model_problems(op: dict, info: dict) -> list[str]:
    """Does a v2 operating point belong to this (dev_all) model? Same family, and
    a training config equivalent to the fold bundles the op was frozen from
    (they must be imported). The identity check of `cli.py export`; a test eval
    runs it before the one-shot look is spent."""
    problems = []
    if op.get("family") != info["family"]:
        problems.append(f"operating point is for family {op.get('family')!r}, the model is "
                        f"{info['family']!r}")
    sources = op.get("sources") or []
    src_infos = [registry.bundle_info(settings.bundles_dir / s["bundle_run"])
                 if isinstance(s, dict) and s.get("bundle_run") else None for s in sources]
    missing = [(s or {}).get("bundle_run") if isinstance(s, dict) else s
               for s, i in zip(sources, src_infos) if i is None]
    if not sources or missing:
        problems.append(f"the operating point's fold bundles are not imported ({missing}) — "
                        "config equivalence cannot be verified (cli.py import-run)")
        return problems
    for s, i in zip(sources, src_infos):  # else the configs compared are another model's
        known = registry.bundle_model_shas(i)
        sha = str(s.get("model_sha256") or "").lower()
        if sha and known and sha not in known:
            problems.append(f"fold bundle {i['run_name']} no longer holds the model the operating "
                            f"point was frozen from ({sha[:12]}…) — stale operating point: "
                            "re-run eval-matrix + choose-op --force")
    return problems + [f"config: {d}" for d in config_differences([info, *src_infos])]


def config_differences(infos: list[dict]) -> list[str]:
    """Human-readable differences in EQUIV_KEYS (+ dataset fingerprint) between bundles."""
    out = []
    sigs = [(i["run_name"], config_signature(i)) for i in infos]
    for key in EQUIV_KEYS:
        values = {json.dumps(sig[key], sort_keys=True, default=str) for _, sig in sigs}
        if len(values) > 1:
            out.append(f"{key} differs: " + ", ".join(f"{n}={sig[key]!r}" for n, sig in sigs))
    fps = {i.get("dataset_fingerprint") for i in infos}
    if len(fps) > 1:
        out.append("dataset_fingerprint differs (different dataset builds): "
                   + ", ".join(f"{i['run_name']}={str(i.get('dataset_fingerprint'))[:12]}" for i in infos))
    return out


def _archive(paths: list[Path], archive_dir: Path) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        if p.exists():
            dest = archive_dir / f"{p.stem}__{stamp}{p.suffix}"
            shutil.move(str(p), str(dest))
            logger.warning(f"archived {p.name} -> {dest}")


def choose_operating_point(
    runs: list[str], target_sensitivity: float = 0.95, out: Path | None = None,
    *, family: str | None = None, force: bool = False, allow_smoke: bool = False,
) -> Path:
    """Pool out-of-fold predictions of a family's fold runs and freeze t*.

    Guards (CONTRACTS §8.3): one run per fold, k DISTINCT fold models, each
    scored on its own validation fold, no smoke runs (unless allow_smoke), one
    family with equivalent training configs. Writes
    ml_data/eval/operating_point_<schema>_<family>.json (+ .md); an existing
    file is frozen — `force` archives it to ml_data/eval/archive/ first."""
    if not valid_threshold(target_sensitivity):
        raise EvalGuardError(f"--target-sensitivity {target_sensitivity!r} must be a number in (0, 1]")
    split = load_split()
    k = len(split["folds"])
    runs = list(runs or [])
    if not runs:
        if not family:
            raise ValueError("choose-op needs the fold eval runs or --family")
        runs = [f"{family}_fold{i}" for i in range(k)]
    elif len(runs) == 1 and not (settings.eval_dir / runs[0] / "predictions.json").exists():
        family = family or runs[0]
        runs = [f"{runs[0]}_fold{i}" for i in range(k)]
    missing = [r for r in runs if not (settings.eval_dir / r / "predictions.json").is_file()]
    if missing:
        raise EvalGuardError(
            f"choose-op: no eval run {missing} in {settings.eval_dir} — score every fold model "
            "on its own fold first (`cli.py eval-matrix`)"
        )
    docs = [load_predictions(r) for r in runs]
    schemas = {d["schema"] for d in docs}
    if len(schemas) != 1:
        raise ValueError(f"runs mix schemas {schemas}")
    schema = schemas.pop()
    buckets = [d["bucket"] for d in docs]
    problems: list[str] = []
    expected = sorted(f"fold{i}" for i in split["folds"])
    if sorted(buckets) != expected:
        problems.append(f"needs exactly one run per fold {expected}, got {buckets}")
    sources = [run_source(d) for d in docs]
    infos = [source_bundle_info(s) for s in sources]
    shas = [s.get("model_sha256") for s in sources]
    unknown = [r for r, s in zip(runs, shas) if not s]
    if unknown:
        problems.append(f"cannot identify the model of {unknown} (model file missing)")
    elif len(set(shas)) != len(shas):
        problems.append(
            "the same model was scored on several folds — pooled predictions are not "
            f"out-of-fold (model sha256s {[s[:12] for s in shas]})"
        )
    for r, d, s, info in zip(runs, docs, sources, infos):
        if d.get("guard_overridden"):
            problems.append(f"{r} was evaluated with guards overridden")
        if info is None:
            problems.append(f"{r}: bundle meta.json of its model not found — cannot verify own-fold")
        elif info["fold"] != d["bucket"]:
            problems.append(
                f"{r}: model {info['run_name']} validates on {info['fold']} but the run "
                f"scored {d['bucket']} (in its training data)"
            )
        elif info["split_id"] != split["split_id"]:
            problems.append(f"{r}: model trained on {info['split_id']!r}, not {split['split_id']!r}")
        if info is not None:  # unconditional: no flag pools a best-on-val model / stale eval
            problems += source_problems(r, s, info)
            if s.get("model_sha256") and not registry.bundle_model_shas(info):
                problems.append(f"{r}: bundle {info['run_name']} has no model.onnx and no "
                                "meta.json onnx.sha256 — cannot verify it is the evaluated model")
    known = [i for i in infos if i]
    families = {i["family"] for i in known}
    if len(families) > 1:
        problems.append(f"runs mix families {sorted(families)}")
    if family and families and families != {family}:
        problems.append(f"--family {family!r} but the runs belong to {sorted(families)}")
    fam = family or (next(iter(families)) if len(families) == 1 else None)
    if fam is None:
        problems.append("cannot determine the family of these runs (give --family)")
    smoke = sorted({n for n in [fam, *runs, *(i["run_name"] for i in known)] if registry.is_smoke(n)})
    if smoke and not allow_smoke:
        problems.append(
            f"smoke runs are plumbing checks, not models ({smoke}) — pass --allow-smoke "
            "to freeze a meaningless test operating point anyway"
        )
    problems += config_differences(known) if len(known) > 1 else []
    if problems:
        raise EvalGuardError("choose-op refused:\n  - " + "\n  - ".join(problems))

    photos = [p for d in docs for p in d["photos"]]
    ids = [p["capture_id"] for p in photos]
    if len(set(ids)) != len(ids):
        raise EvalGuardError("the fold runs overlap (a capture appears twice) — not a partition")
    t = threshold_for_sensitivity(photos, target_sensitivity)
    if t is None:
        raise RuntimeError("no positive photos with a detection — cannot set t*")
    cut = [f"{r} (conf_floor {s['conf_floor']})" for r, s in zip(runs, sources)
           if isinstance(s.get("conf_floor"), (int, float)) and s["conf_floor"] > t]
    if cut:  # scores between t* and their floor were never stored: t* is not reproducible
        raise EvalGuardError(
            f"choose-op refused: t*={t:.6g} lies below the confidence floor of {cut} — their "
            "detections under the floor were cut off; re-evaluate with a lower --conf"
        )
    metrics = score_all(
        photos, docs[0]["class_names"], t,
        threshold_source=f"pooled out-of-fold {buckets} (t* fitted here → mildly optimistic)",
        probe=[p for p in split["probe_boxes"] if p["class"] in docs[0]["class_names"]
               and p["capture_id"] in set(ids)],
    )
    op = {
        "schema": schema,
        "class_names": docs[0]["class_names"],
        "threshold": t,
        "rule": f"highest t with pooled out-of-fold photo sensitivity >= {target_sensitivity}",
        "runs": runs,
        "models": [d["model"] for d in docs],
        "split_id": split["split_id"],
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "oof_level1": metrics["level1_verdict"],
        "family": fam,
        "target_sensitivity": target_sensitivity,
        "allow_smoke": bool(smoke),  # frozen from smoke runs under --allow-smoke: plumbing only
        "sources": [
            {"eval_run": r, "bundle_run": i["run_name"], "fold": i["fold"],
             "model_sha256": s["model_sha256"], "config_sha256": i["config_sha256"],
             "dataset_fingerprint": i["dataset_fingerprint"]}
            for r, s, i in zip(runs, sources, infos)
        ],
    }
    out = Path(out) if out else settings.eval_dir / f"operating_point_{schema}_{fam}.json"
    md = out.with_suffix(".md")
    if out.exists():
        if not force:
            raise EvalGuardError(
                f"{out.name} exists and is FROZEN (the inspector/export may depend on it) — "
                "pass --force to archive it and write a new one"
            )
        _archive([out, md], settings.eval_dir / "archive")
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(out, json.dumps(op, indent=1, ensure_ascii=False))
    pooled = {**metrics, "run_name": f"OOF[{', '.join(runs)}]", "schema": schema,
              "bucket": "oof_dev", "n_photos": len(photos), "model": "(pooled fold models)"}
    report = format_report(pooled)
    header = [f"_Operating point `{out.name}` — family `{fam}`, target sensitivity "
              f"{target_sensitivity}, t* = {t!r}._", ""]
    header += [f"- {s['eval_run']}: bundle `{s['bundle_run']}` ({s['fold']}), model "
               f"`{s['model_sha256'][:12]}`" for s in op["sources"]]
    if smoke:
        header.append("- ⚠ **smoke runs (allow_smoke)** — plumbing only, not a real operating point")
    _atomic_write_text(md, "\n".join(header) + "\n\n" + report)
    logger.info("\n" + report)
    logger.info(f"operating point frozen: t*={t:.6g} -> {out}")
    return out


# ----------------------------------------------------------------- outputs


def _fmt_rate(r: dict | None) -> str:
    if not r or r.get("rate") is None:
        return "–"
    ci = r["ci95"]
    return f"{r['rate']:.3f} ({r['k']}/{r['n']}; CI {ci[0]:.2f}–{ci[1]:.2f})"


def format_report(m: dict) -> str:
    names = m["class_names"]
    l1, l3 = m["level1_verdict"], m["level3_defects"]
    t = m["threshold"]
    lines = [
        f"# Eval {m['run_name']}",
        "",
        f"- model `{Path(str(m['model'])).name}` · schema `{m['schema']}` ({', '.join(names)}) · "
        f"bucket `{m['bucket']}` · {m['n_photos']} photos",
        f"- threshold t = {t if t is None else round(t, 4)} — {m['threshold_source']}",
    ]
    src = m.get("source") or {}
    if src.get("model_sha256"):
        lines.append(
            f"- {src.get('kind', 'detector')} · bundle `{src.get('bundle_run')}` "
            f"({src.get('fold')}) · model sha256 `{src['model_sha256'][:12]}`"
        )
    if m.get("guard_overridden"):
        lines.append("- ⚠ **guard overridden — not an honest number**: "
                     + "; ".join(m["guard_overridden"]))
    tl = m.get("test_look")
    if tl:
        lines.append(f"- test look: **{tl['look']}**" + (f" — {tl['reason']}" if tl.get("reason") else "")
                     + f" (recorded in reports/{tl.get('ledger') or 'test_ledger.csv'})")
    if m["bucket"] in NOT_HONEST_BUCKETS:
        lines.append(f"- ⚠ {NOT_HONEST_BUCKETS[m['bucket']]}")
    lines += [
        "",
        "## 1 · Verdict (i.O / N.iO) — acceptance level",
        "",
        f"- positives {l1['n_positive']} · negatives (iO) {l1['n_negative']} · "
        f"other-class-only (not scored) {l1['n_other_only']}",
        f"- photo AUROC {l1['photo_auroc']} (seated only: {l1['photo_auroc_seated']})",
        f"- **escape rate** {_fmt_rate(l1.get('escape'))}",
        f"- **false-call rate, seated iO** {_fmt_rate(l1.get('false_call_seated'))}",
        f"- false-call rate, handheld iO {_fmt_rate(l1.get('false_call_handheld'))}",
        f"- share of flagged photos that are iO {_fmt_rate(l1.get('false_call_share_of_flagged'))}",
    ]
    if l1.get("by_board_pose"):
        lines += ["", "| board/pose | escape | false call |", "|---|---|---|"]
        for k, v in l1["by_board_pose"].items():
            lines.append(f"| {k} | {_fmt_rate(v['escape'])} | {_fmt_rate(v['false_call'])} |")
    if "B" in {k.split('/')[0] for k in l1.get("by_board_pose", {})}:
        lines.append("\n_Board B rows = within-panel (memorisation) — board B is always trained._")

    l2 = m.get("level2_classes")
    if l2:
        lines += ["", "## 2 · Which error — class level", "",
                  "| class | photo sensitivity | class false alarm | reported on iO photos, by board |",
                  "|---|---|---|---|"]
        for name, v in l2["per_class"].items():
            by_board = ", ".join(
                f"{b}: {r['k']}/{r['n']}" for b, r in v["false_alarm_on_iO_by_board"].items()
            ) or "–"
            lines.append(f"| {SHORT.get(name, name)} | {_fmt_rate(v['photo_sensitivity'])} | "
                         f"{_fmt_rate(v['class_false_alarm'])} | {by_board} |")
        lines += ["", f"- exact class-set accuracy on N.iO photos {_fmt_rate(l2['exact_set_accuracy'])}",
                  "", "Photo-level class confusion (rows = true class present; cols = classes reported):", ""]
        cols = names + ["(none)"]
        lines += ["| true \\ reported | " + " | ".join(SHORT.get(c, c) for c in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for row, counts in l2["photo_class_confusion"].items():
            lines.append(f"| {SHORT.get(row, row)} | " + " | ".join(str(counts.get(c, 0)) for c in cols) + " |")

    lines += ["", f"## 3 · Defect level — model selection (recall @ {FPPI_OPERATING} FPPI, IoU 0.3)", "",
              "| class | GT boxes | recall@0.5FPPI (CI, unit) | AP50 | FP on other-class defects |",
              "|---|---|---|---|---|"]
    for name, v in list(l3["per_class"].items()) + [("any class", l3["class_agnostic"])]:
        if not v.get("n_gt"):
            lines.append(f"| {SHORT.get(name, name)} | 0 | – | – | – |")
            continue
        lines.append(
            f"| {SHORT.get(name, name)} | {v['n_gt']} | {v['recall_at_op']:.3f} "
            f"({v['recall_ci95'][0]:.2f}–{v['recall_ci95'][1]:.2f}, {v['ci_unit']}) | "
            f"{v['ap50']:.3f} | {v['fp_at_op_other_defect']} |"
        )
    bc = l3.get("box_confusion")
    if bc:
        lines += ["", f"Box-level confusion at t (rows = GT, cols = predicted):", "",
                  "| GT \\ pred | " + " | ".join(SHORT.get(c, c) for c in bc["columns"]) + " |",
                  "|---|" + "---|" * len(bc["columns"])]
        for row, counts in bc["rows"].items():
            lines.append(f"| {SHORT.get(row, row)} | " + " | ".join(str(counts.get(c, 0)) for c in bc["columns"]) + " |")

    probe = m.get("probe_xprod")
    if probe and probe["boxes"]:
        lines += ["", "## Cross-product probe (board-A/A2 Brücke/Fahne, never trained)", ""]
        for cls, s in probe["summary"].items():
            lines.append(f"- **{cls}**: found {s['found']}/{s['n']}, correct class {s['class_correct']}/{s['n']}")
        lines += ["", "_Counts, not rates — the only honest cross-product evidence available._"]
    if "mean_ms_per_photo" in m:
        lines += ["", f"- mean {m['mean_ms_per_photo']} ms/photo on {m.get('host')}"]
    return "\n".join(lines) + "\n"


def _gallery(out_dir: Path, m: dict, photos: list[dict]) -> None:
    t = m["threshold"]
    names = m["class_names"]
    manifest = {e["capture_id"]: e for e in load_manifest()["entries"]}
    fn_items, fp_items, conf_items = [], [], []
    for p in photos:
        keep = [i for i, s in enumerate(p["scores"]) if t is not None and s >= t]
        boxes = [p["boxes"][i] for i in keep]
        scores = [p["scores"][i] for i in keep]
        labels = [p["labels"][i] for i in keep]
        g2p, matched = match_indices(boxes, scores, [b for _, b in p["gt"]], IOU_OPERATING)
        for (c, b), j in zip(p["gt"], g2p):
            if j is None:
                fn_items.append({"capture_id": p["capture_id"], "box": b,
                                 "label": f"missed {SHORT.get(names[c], names[c])}"})
            elif labels[j] != c:
                conf_items.append({"capture_id": p["capture_id"], "box": b,
                                   "label": f"{SHORT.get(names[c], names[c])} → "
                                            f"{SHORT.get(names[labels[j]], names[labels[j]])} "
                                            f"{scores[j]:.2f}"})
        for j, ok in enumerate(matched):
            if not ok:
                other = any(iou(boxes[j], b) >= 0.1 for _, b in p["other_gt"])
                fp_items.append({"capture_id": p["capture_id"], "box": boxes[j], "score": scores[j],
                                 "label": f"FP {SHORT.get(names[labels[j]], names[labels[j]])} "
                                          f"{scores[j]:.2f}" + (" · on other defect" if other else "")})
    fp_items.sort(key=lambda x: -x["score"])
    write_gallery(
        out_dir / "gallery.html",
        {cid: photo_path(manifest[cid]) for cid in {p["capture_id"] for p in photos}},
        {"Missed defects": fn_items, "Wrong class": conf_items, "False positives": fp_items},
        f"{m['run_name']} — errors at t={t if t is None else round(t, 3)}",
    )


def results_path() -> Path:
    return settings.reports_dir / RESULTS_FILE


def _append_results_row(m: dict) -> None:
    if m["bucket"] in NOT_RECORDED_BUCKETS:
        logger.info(f"not recorded in {RESULTS_FILE}: {m['bucket']} is a memorisation check")
        return
    l1, l3 = m["level1_verdict"], m["level3_defects"]
    names = m["class_names"]

    def rec(name):
        v = l3["per_class"].get(name, {})
        return v.get("recall_at_op") if v.get("n_gt") else None

    nal = l3["per_class"].get(names[0], {})
    esc = l1.get("escape") or {}
    fcs = l1.get("false_call_seated") or {}
    probe = (m.get("probe_xprod") or {}).get("summary", {})
    row = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "run_name": m["run_name"], "model": Path(str(m["model"])).name,
        "adapter": m["adapter"], "schema": m["schema"], "bucket": m["bucket"],
        "threshold": m["threshold"], "threshold_source": m["threshold_source"][:60],
        "n_photos": m["n_photos"], "n_positive": l1["n_positive"],
        "n_negative_seated": (fcs or {}).get("n"),
        "photo_auroc": l1["photo_auroc"],
        "escape": esc.get("rate"), "escape_ci_lo": (esc.get("ci95") or [None, None])[0],
        "escape_ci_hi": (esc.get("ci95") or [None, None])[1],
        "fcr_seated": fcs.get("rate"), "fcr_seated_ci_lo": (fcs.get("ci95") or [None, None])[0],
        "fcr_seated_ci_hi": (fcs.get("ci95") or [None, None])[1],
        "fcr_handheld": (l1.get("false_call_handheld") or {}).get("rate"),
        "recall_nal_0.5fppi": rec("Nicht ausreichend Lot"),
        "recall_nal_ci_lo": (nal.get("recall_ci95") or [None, None])[0],
        "recall_nal_ci_hi": (nal.get("recall_ci95") or [None, None])[1],
        "recall_bruecke_0.5fppi": rec("Brücke"), "recall_fahne_0.5fppi": rec("Fahne"),
        "recall_any_0.5fppi": l3["class_agnostic"].get("recall_at_op"),
        "exact_set_acc": ((m.get("level2_classes") or {}).get("exact_set_accuracy") or {}).get("rate"),
        "ap50_nal": nal.get("ap50"),
        "probe_found": "; ".join(f"{c} {s['found']}/{s['n']}" for c, s in probe.items()) or None,
        "mean_ms_per_photo": m["mean_ms_per_photo"], "tiler_version": m["tiler_version"],
        "split_id": m["split_id"], "snapshot_id": m["snapshot_id"],
        "dataset_fingerprint": (m.get("dataset_fingerprint") or "")[:16] or None,
        "host": m["host"],
    }
    if m["bucket"] in PROBE_ONLY_BUCKETS:
        row.update({k: None for k in _NOT_HONEST_ON_PROBE})
    path = results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
