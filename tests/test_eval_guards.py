"""
Evaluation honesty guards (ml/eval/run_eval.py, docs/CONTRACTS.md §8.2).

Every refusal must fire BEFORE a photo is read or an ONNX session is created:
here the fake "models" are not even valid ONNX, and TiledDetector /
bucket_photos are replaced by tripwires. probe_xprod is checked against the
real committed split. predictions.json v2 writing + v1 loading are covered too.

Run: ml/.venv/bin/python -B tests/test_eval_guards.py
"""
import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="zander_eval_guards_"))
os.environ["ML_DATA_ROOT"] = str(TMP)  # BEFORE config.settings is imported
sys.path.insert(0, str(ROOT / "ml"))

import registry  # noqa: E402
from config.settings import settings  # noqa: E402
from dataprep.ingest import load_manifest  # noqa: E402
from dataprep.split import bucket_of, load_split  # noqa: E402
from eval import run_eval  # noqa: E402
from eval.photo_scorer import score_all  # noqa: E402
from eval.run_eval import EvalGuardError  # noqa: E402

NAMES = ["Nicht ausreichend Lot", "Brücke", "Fahne"]
SPLIT = load_split()
REAL_BUCKET_PHOTOS = run_eval.bucket_photos
RESULTS = ROOT / "ml" / "reports" / "results_v2.csv"
RESULTS_BEFORE = RESULTS.read_bytes() if RESULTS.exists() else None


class Reached(Exception):
    """Tripwire: code after the guards was reached."""


def _no_onnx(*a, **k):
    raise AssertionError("an ONNX session was created before the guards ran")


def _no_photos(*a, **k):
    raise Reached("bucket_photos reached")


run_eval.TiledDetector = _no_onnx
run_eval.bucket_photos = _no_photos
run_eval.test_ledger_path = lambda: TMP / "ledger_default.csv"  # never the committed one
assert settings.ml_data_root == TMP


def _bundle(name: str, **meta_over) -> Path:
    d = TMP / "bundles" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.onnx").write_bytes(b"not an onnx model: " + name.encode())
    meta = {"run_name": name, "framework": "ultralytics", "adapter": "ultralytics",
            "fold": "fold0", "schema": "def3", "classes": NAMES,
            "split_id": SPLIT["split_id"], "dataset_id": "ds_v2_def3", "tiler_version": "v2"}
    meta.update(meta_over)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d / "model.onnx"


def _op(name: str = "op.json", threshold: float = 0.3) -> Path:
    p = TMP / name
    p.write_text(json.dumps({"schema": "def3", "class_names": NAMES, "threshold": threshold,
                             "split_id": SPLIT["split_id"]}), encoding="utf-8")
    return p


def _family_op(family: str, name: str | None = None, **over) -> Path:
    """Fold bundles of `family` + a v2 operating point frozen from them (choose-op's shape)."""
    _fold_family(family)
    sources = []
    for k in range(3):
        info = registry.bundle_info(TMP / "bundles" / f"{family}_fold{k}")
        sources.append({"eval_run": f"{family}_fold{k}", "bundle_run": info["run_name"],
                        "fold": f"fold{k}", "model_sha256": registry.file_sha256(info["model_path"]),
                        "config_sha256": info["config_sha256"], "dataset_fingerprint": None})
    op = {"schema": "def3", "class_names": NAMES, "threshold": 0.3, "split_id": SPLIT["split_id"],
          "rule": "highest t with pooled out-of-fold photo sensitivity >= 0.95",
          "runs": [s["eval_run"] for s in sources], "family": family, "target_sensitivity": 0.95,
          "sources": sources}
    op.update(over)
    p = TMP / (name or f"op_{family}.json")
    p.write_text(json.dumps(op), encoding="utf-8")
    return p


def _devall(run: str, family: str, model_bytes: bytes | None = None, epochs=120) -> Path:
    """A dev_all bundle of `family`, equivalent to _fold_family's recipe (extra.val differs only)."""
    model = _bundle(run, fold="dev_all", family=family)
    if model_bytes:
        model.write_bytes(model_bytes)
    _config(model, run, "dev_all", epochs, extra={"val": False})
    return model


def _raises(fn, exc, *needles):
    try:
        fn()
    except exc as e:
        for n in needles:
            assert n in str(e), f"{n!r} not in: {e}"
        return e
    raise AssertionError(f"expected {exc.__name__}")


def _ledger(tmpname: str) -> Path:
    path = TMP / tmpname
    run_eval.test_ledger_path = lambda: path
    return path


def test_fold_bucket_needs_the_models_own_fold():
    model = _bundle("m_def3_fold0")
    _raises(lambda: run_eval.run(model, "ultralytics", "def3", "fold1"), EvalGuardError,
            "validates on fold0", "--allow-foreign-fold")
    # its own fold passes every guard (and only then reaches the photos)
    _raises(lambda: run_eval.run(model, "ultralytics", "def3", "fold0"), Reached)


def test_dev_all_model_refused_on_every_fold():
    model = _bundle("m_def3_devall", fold="dev_all")
    for bucket in ("fold0", "fold1", "fold2"):
        _raises(lambda: run_eval.run(model, "ultralytics", "def3", bucket), EvalGuardError, "dev_all")


def test_missing_meta_refused_unless_overridden():
    d = TMP / "loose"
    d.mkdir(exist_ok=True)
    (d / "model.onnx").write_bytes(b"x")
    _raises(lambda: run_eval.run(d / "model.onnx", "ultralytics", "def3", "fold0"),
            EvalGuardError, "no meta.json")
    info, overridden = run_eval.check_eval_guards(d / "model.onnx", "ultralytics", "def3", "fold0",
                                                  allow_foreign_fold=True)
    assert info is None and overridden and "no meta.json" in overridden[0]


def test_override_is_stamped_not_silent():
    model = _bundle("m2_def3_fold0")
    info, overridden = run_eval.check_eval_guards(model, "ultralytics", "def3", "fold2",
                                                  allow_foreign_fold=True)
    assert info["fold"] == "fold0" and any("fold2" in o for o in overridden)


def test_schema_adapter_split_mismatch_is_never_overridable():
    nal = _bundle("m_nal1_fold0", schema="nal1", classes=NAMES[:1])
    _raises(lambda: run_eval.run(nal, "ultralytics", "def3", "fold0", allow_foreign_fold=True),
            EvalGuardError, "schema")
    old = _bundle("m_old_fold0", split_id="split_v1")
    _raises(lambda: run_eval.run(old, "ultralytics", "def3", "fold0", allow_foreign_fold=True),
            EvalGuardError, "split_v1")
    deim = _bundle("m_deim_fold0", adapter="deim", framework="deim")
    _raises(lambda: run_eval.run(deim, "ultralytics", "def3", "fold0", allow_foreign_fold=True),
            EvalGuardError, "adapter")


def test_unknown_bucket_refused():
    model = _bundle("m3_def3_fold0")
    for bucket in ("dev", "b_holdout", "fold3"):
        _raises(lambda: run_eval.run(model, "ultralytics", "def3", bucket), EvalGuardError,
                "unknown bucket")


def test_frozen_buckets_need_an_operating_point():
    model = _bundle("m4_def3_devall", fold="dev_all")
    for bucket in ("test", "probe_xprod"):
        _raises(lambda: run_eval.run(model, "ultralytics", "def3", bucket), EvalGuardError,
                "--operating-point")


def test_fold_model_on_test_refused():
    _ledger("ledger_fold_on_test.csv")
    model = _bundle("m5_def3_fold1", fold="fold1")
    _raises(lambda: run_eval.run(model, "ultralytics", "def3", "test", operating_point=_op()),
            EvalGuardError, "dev_all")


def test_test_ledger_one_shot_rule():
    path = _ledger("ledger_rule.csv")
    first = run_eval.check_test_ledger("a" * 64, "b" * 64)
    assert first["look"] == "first"
    run_eval.record_test_look(first, run_name="x__test", model_sha="a" * 64, op_sha="b" * 64,
                              bundle_run="x", family="f", schema="def3", threshold=0.3)
    assert run_eval.check_test_ledger("a" * 64, "b" * 64)["look"] == "rerun"
    _raises(lambda: run_eval.check_test_ledger("c" * 64, "b" * 64), EvalGuardError,
            "second look", "--second-look")
    _raises(lambda: run_eval.check_test_ledger("a" * 64, "d" * 64), EvalGuardError, "second look")
    second = run_eval.check_test_ledger("c" * 64, "b" * 64, second_look="  new dev_all retrain  ")
    assert second == {"look": "second_look", "reason": "new dev_all retrain"}
    run_eval.record_test_look(second, run_name="y__test", model_sha="c" * 64, op_sha="b" * 64,
                              bundle_run="y", family="f", schema="def3", threshold=0.3)
    run_eval.record_test_look({"look": "rerun", "reason": ""}, run_name="y__test", model_sha="c" * 64,
                              op_sha="b" * 64, bundle_run="y", family="f", schema="def3", threshold=0.3)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["look"] for r in rows] == ["first", "second_look"], rows
    assert rows[1]["reason"] == "new dev_all retrain"
    assert list(rows[0]) == run_eval.TEST_LEDGER_FIELDS


def test_committed_ledger_header_matches_the_code():
    with open(ROOT / "ml" / "reports" / "test_ledger.csv", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == run_eval.TEST_LEDGER_FIELDS, header


def test_different_model_on_test_raises_via_the_ledger():
    _ledger("ledger_e2e.csv")
    op = _family_op("ma_def3", "op_e2e.json")
    model_a = _devall("ma_def3_devall", "ma_def3")
    model_b = _devall("ma_def3_devall_retrain", "ma_def3", b"a retrained dev_all model")
    look = run_eval.check_test_ledger(registry.file_sha256(model_a), registry.file_sha256(op))
    run_eval.record_test_look(look, run_name="ma_def3_devall__test",
                              model_sha=registry.file_sha256(model_a), op_sha=registry.file_sha256(op),
                              bundle_run="ma_def3_devall", family="ma_def3", schema="def3",
                              threshold=0.3)
    _raises(lambda: run_eval.run(model_b, "ultralytics", "def3", "test", operating_point=op),
            EvalGuardError, "second look")
    # the recorded pair may re-run: the guards pass and the photos are reached
    _raises(lambda: run_eval.run(model_a, "ultralytics", "def3", "test", operating_point=op), Reached)
    # a stated second look passes the guards too
    _raises(lambda: run_eval.run(model_b, "ultralytics", "def3", "test", operating_point=op,
                                 second_look="retrained after label fix"), Reached)


def test_probe_xprod_has_no_test_photo():
    entries, probe = REAL_BUCKET_PHOTOS("probe_xprod", "def3")
    manifest = load_manifest()
    by_id = {e["capture_id"]: e for e in manifest["entries"]}
    buckets = {bucket_of(e, SPLIT) for e in entries}
    assert "test" not in buckets and entries, buckets
    probe_test = {p["capture_id"] for p in SPLIT["probe_boxes"]
                  if bucket_of(by_id[p["capture_id"]], SPLIT) == "test"}
    assert probe_test, "fixture assumption: the split has a probe photo in test"
    assert not probe_test & {e["capture_id"] for e in entries}
    assert {p["capture_id"] for p in probe} <= {e["capture_id"] for e in entries}
    non_test = {p["capture_id"] for p in SPLIT["probe_boxes"]} - probe_test
    assert non_test == {e["capture_id"] for e in entries}


def test_default_run_names_follow_the_contract():
    fold_model = _bundle("yolo_x_def3_fold1", fold="fold1")
    info = registry.bundle_info(fold_model.parent)
    assert run_eval.default_run_name(info, fold_model, "def3", "fold1") == "yolo_x_def3_fold1"
    assert run_eval.default_run_name(info, fold_model, "def3", "probe_xprod") == \
        "yolo_x_def3_fold1__probe_xprod"
    dev = _bundle("yolo_x_def3_devall", fold="dev_all")
    assert run_eval.default_run_name(registry.bundle_info(dev.parent), dev, "def3", "test") == \
        "yolo_x_def3_devall__test"


def _photos():
    return [
        {"capture_id": "p1", "group": "g1", "board": "A", "pose": "seated", "gt_verdict": "NiO",
         "width": 2616, "height": 1960, "gt": [(0, (10.0, 10.0, 40.0, 40.0))], "other_gt": [],
         "boxes": [(10.04, 10.0, 40.0, 40.0)], "scores": [0.123456789012], "labels": [0]},
        {"capture_id": "p2", "group": "g2", "board": "A", "pose": "seated", "gt_verdict": "iO",
         "width": 2616, "height": 1960, "gt": [], "other_gt": [("Brücke", (5.0, 5.0, 9.0, 9.0))],
         "boxes": [], "scores": [], "labels": []},
    ]


def test_predictions_v2_written_and_v1_still_loads():
    photos = _photos()
    metrics = score_all(photos, NAMES, 0.1, "explicit")
    metrics.update({"run_name": "w_run", "model": "m.onnx", "adapter": "ultralytics",
                    "schema": "def3", "bucket": "fold0", "n_photos": 2, "mean_ms_per_photo": 1.0,
                    "tiler_version": "v2", "split_id": "split_v2", "snapshot_id": "s",
                    "dataset_fingerprint": None, "host": "h"})
    source = {"kind": "detector", "bundle_run": "b", "family": "f", "fold": "fold0",
              "model_sha256": "e" * 64, "model_resolved": "/x/m.onnx", "adapter": "ultralytics",
              "conf_floor": 0.001, "verifier_run": None, "stacker_sha256": None}
    out = settings.eval_dir / "w_run"
    run_eval.write_outputs(out, metrics, photos, record=False, gallery=False, source=source)
    assert not (out / "gallery.html").exists()
    raw = json.loads((out / "predictions.json").read_text(encoding="utf-8"))
    assert raw["format_version"] == 2 and raw["source"] == source
    assert raw["photos"][0]["width"] == 2616 and raw["photos"][0]["height"] == 1960
    assert raw["photos"][0]["scores"] == [0.123456789012], "scores must keep full precision"
    assert (out / "metrics.json").is_file() and (out / "report.md").is_file()
    doc = run_eval.load_predictions("w_run")
    assert doc["photos"][0]["gt"] == [(0, (10.0, 10.0, 40.0, 40.0))]
    # a v1 file (no format_version / source / width) still loads
    v1 = settings.eval_dir / "v1_run"
    v1.mkdir(parents=True, exist_ok=True)
    del raw["format_version"], raw["source"]
    for p in raw["photos"]:
        del p["width"], p["height"]
    (v1 / "predictions.json").write_text(json.dumps(raw), encoding="utf-8")
    old = run_eval.load_predictions("v1_run")
    assert old["format_version"] == 1 and old["source"] is None
    assert old["photos"][0]["boxes"] == [(10.0, 10.0, 40.0, 40.0)]
    # the pre-v2 list format is refused with a clear message
    (settings.eval_dir / "v0_run").mkdir()
    (settings.eval_dir / "v0_run" / "predictions.json").write_text("[]", encoding="utf-8")
    _raises(lambda: run_eval.load_predictions("v0_run"), ValueError, "re-evaluate")


def test_real_smoke_v1_predictions_load():
    src = ROOT / "ml_data" / "eval" / "smoke_def3_fold0" / "predictions.json"
    if not src.is_file():
        print("SKIP real v1 predictions (ml_data/eval/smoke_def3_fold0 absent)")
        return
    dest = settings.eval_dir / "smoke_copy"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "predictions.json").write_bytes(src.read_bytes())
    doc = run_eval.load_predictions("smoke_copy")
    assert doc["format_version"] == 1 and doc["schema"] == "def3" and doc["photos"]


def _config(model: Path, run: str, fold: str, epochs=120, extra=None, batch=16) -> None:
    (model.parent / "config.yaml").write_text(
        f"name: {run}\nmodel: yolo26n.pt\ndataset_id: ds_v2_def3\nfold: {fold}\n"
        f"epochs: {epochs}\nimgsz: 640\nbatch: {batch}\n"
        + (f"extra: {json.dumps(extra)}\n" if extra is not None else ""), encoding="utf-8")


def _fold_family(family: str, pos_scores=(0.9, 0.7, 0.4), same_model=False, epochs=(120, 120, 120),
                 fold_meta=({}, {}, {}), fold_extra=(None, None, None)):
    """3 fold bundles (own meta/config/model) + their own-fold eval runs (v2)."""
    for k in range(3):
        run = f"{family}_fold{k}"
        model = _bundle(run, fold=f"fold{k}", family=family, **fold_meta[k])
        model.write_bytes(b"one shared model" if same_model else b"model of " + run.encode())
        _config(model, run, f"fold{k}", epochs[k], fold_extra[k])
        photos = [
            {"capture_id": f"{family}_pos{k}", "group": f"g{k}a", "board": "A", "pose": "seated",
             "gt_verdict": "NiO", "gt": [[0, [10, 10, 40, 40]]], "other_gt": [],
             "boxes": [[10, 10, 40, 40]], "scores": [pos_scores[k]], "labels": [0]},
            {"capture_id": f"{family}_neg{k}", "group": f"g{k}b", "board": "A", "pose": "seated",
             "gt_verdict": "iO", "gt": [], "other_gt": [],
             "boxes": [[50, 50, 70, 70]], "scores": [0.05 * (k + 1)], "labels": [0]},
        ]
        doc = {"format_version": 2, "schema": "def3", "class_names": NAMES, "bucket": f"fold{k}",
               "model": str(model),
               "source": {"kind": "detector", "bundle_run": run, "family": family, "fold": f"fold{k}",
                          "model_sha256": registry.file_sha256(model), "model_resolved": str(model),
                          "adapter": "ultralytics", "conf_floor": 0.001, "verifier_run": None,
                          "stacker_sha256": None},
               "photos": photos}
        d = settings.eval_dir / run
        d.mkdir(parents=True, exist_ok=True)
        (d / "predictions.json").write_text(json.dumps(doc), encoding="utf-8")


def test_choose_op_freezes_a_v2_operating_point():
    _fold_family("cf_def3")
    out = run_eval.choose_operating_point([], family="cf_def3")
    assert out == settings.eval_dir / "operating_point_def3_cf_def3.json" and out.with_suffix(".md").is_file()
    op = json.loads(out.read_text(encoding="utf-8"))
    for key in ("schema", "class_names", "threshold", "rule", "runs", "models", "split_id",
                "created", "oof_level1", "family", "target_sensitivity", "sources"):
        assert key in op, key
    assert op["threshold"] == 0.4 and op["family"] == "cf_def3" and op["target_sensitivity"] == 0.95
    assert op["runs"] == ["cf_def3_fold0", "cf_def3_fold1", "cf_def3_fold2"]
    assert [s["fold"] for s in op["sources"]] == ["fold0", "fold1", "fold2"]
    assert len({s["model_sha256"] for s in op["sources"]}) == 3
    assert all(s["config_sha256"] and s["dataset_fingerprint"] is None for s in op["sources"])
    # frozen: a second choose-op is refused; --force archives the old file first
    _raises(lambda: run_eval.choose_operating_point(["cf_def3"]), EvalGuardError, "FROZEN", "--force")
    run_eval.choose_operating_point(["cf_def3"], force=True)
    archived = sorted(p.name for p in (settings.eval_dir / "archive").iterdir())
    assert len(archived) == 2 and all(n.startswith("operating_point_def3_cf_def3__") for n in archived)


def test_choose_op_refuses_non_oof_or_mixed_recipes():
    _fold_family("same_def3", same_model=True)
    _raises(lambda: run_eval.choose_operating_point([], family="same_def3"), EvalGuardError,
            "same model")
    _fold_family("drift_def3", epochs=(120, 120, 60))
    _raises(lambda: run_eval.choose_operating_point([], family="drift_def3"), EvalGuardError,
            "epochs differs")
    _fold_family("x_smoke_def3")
    _raises(lambda: run_eval.choose_operating_point([], family="x_smoke_def3"), EvalGuardError,
            "--allow-smoke")
    run_eval.choose_operating_point([], family="x_smoke_def3", allow_smoke=True)
    _raises(lambda: run_eval.choose_operating_point(["cf_def3_fold0", "cf_def3_fold0", "cf_def3_fold2"],
                                                    out=settings.eval_dir / "x.json"),
            EvalGuardError, "one run per fold")
    _raises(lambda: run_eval.choose_operating_point([], family="nope_def3"), EvalGuardError,
            "eval-matrix")


def test_training_policy_violations_are_honesty_guards():
    """weights best / early stopping / a truncated run: refused, overridable + stamped."""
    best = _bundle("pol_best_def3_fold0", weights="best")
    _raises(lambda: run_eval.run(best, "ultralytics", "def3", "fold0"), EvalGuardError,
            "weights 'best'", "--allow-foreign-fold")
    _, overridden = run_eval.check_eval_guards(best, "ultralytics", "def3", "fold0",
                                               allow_foreign_fold=True)
    assert any("weights 'best'" in o for o in overridden), overridden
    pat = _bundle("pol_pat_def3_fold0")
    _config(pat, "pol_pat_def3_fold0", "fold0", extra={"patience": 20})
    _raises(lambda: run_eval.run(pat, "ultralytics", "def3", "fold0"), EvalGuardError,
            "extra.patience=20")
    stop = _bundle("pol_stop_def3_fold0", epochs=120, epochs_completed=61)
    _raises(lambda: run_eval.run(stop, "ultralytics", "def3", "fold0"), EvalGuardError,
            "stopped after 61 of 120")
    dev = _bundle("pol_best_def3_devall", fold="dev_all", weights="best")
    _raises(lambda: run_eval.run(dev, "ultralytics", "def3", "test", operating_point=_op()),
            EvalGuardError, "weights 'best'")
    # the run policy passes: last epoch, patience 0, every epoch trained ...
    ok = _bundle("pol_ok_def3_fold0", weights="last", epochs=120, epochs_completed=120)
    _config(ok, "pol_ok_def3_fold0", "fold0", extra={"patience": 0})
    _raises(lambda: run_eval.run(ok, "ultralytics", "def3", "fold0"), Reached)
    # ... and so does a DEIM bundle (no extra.patience, always last)
    deim = _bundle("pol_deim_def3_fold0", framework="deim", adapter="deim", weights="last",
                   epochs=80, epochs_completed=80)
    _config(deim, "pol_deim_def3_fold0", "fold0", epochs=80, extra={})
    _raises(lambda: run_eval.run(deim, "deim", "def3", "fold0"), Reached)


def test_choose_op_refuses_selected_or_early_stopped_fold_models():
    """Unconditional: not even --allow-smoke pools a best-on-val / early-stopped model."""
    _fold_family("pol_smoke_def3", fold_meta=({}, {"weights": "best"}, {}))
    _raises(lambda: run_eval.choose_operating_point([], family="pol_smoke_def3", allow_smoke=True),
            EvalGuardError, "pol_smoke_def3_fold1", "weights 'best'")
    _fold_family("pol2_def3", fold_extra=(None, None, {"patience": 20}))
    _raises(lambda: run_eval.choose_operating_point([], family="pol2_def3"), EvalGuardError,
            "pol2_def3_fold2", "patience")
    _fold_family("pol3_def3", fold_meta=({"epochs_completed": 50}, {}, {}))
    _raises(lambda: run_eval.choose_operating_point([], family="pol3_def3"), EvalGuardError,
            "stopped after 50 of 120")


def test_operating_point_threshold_is_validated():
    """A null threshold made a frozen-op bucket self-tune on itself (score_all(None))."""
    _ledger("ledger_op_validation.csv")
    model = _bundle("opv_def3_devall", fold="dev_all")
    for bad in (None, True, 0, -0.1, 1.5, float("nan"), "0.3"):
        _raises(lambda: run_eval.run(model, "ultralytics", "def3", "probe_xprod",
                                     operating_point=_op("op_bad.json", threshold=bad)),
                EvalGuardError, "threshold")
    _raises(lambda: run_eval.run(model, "ultralytics", "def3", "test",
                                 operating_point=_family_op("opv_def3", "op_null.json", threshold=None)),
            EvalGuardError, "threshold")
    # a hand-written / v1 op is fine for probe_xprod, never for the one-shot test
    _raises(lambda: run_eval.run(model, "ultralytics", "def3", "probe_xprod",
                                 operating_point=_op("op_v1.json", threshold=1.0)), Reached)
    _raises(lambda: run_eval.run(model, "ultralytics", "def3", "test", operating_point=_op("op_v1.json")),
            EvalGuardError, "choose-op provenance", "family")
    # --conf above the frozen threshold would cut detections off
    _raises(lambda: run_eval.run(model, "ultralytics", "def3", "probe_xprod",
                                 operating_point=_op("op_lo.json", threshold=0.3), conf_floor=0.5),
            EvalGuardError, "--conf 0.5")


def test_test_eval_checks_the_op_belongs_to_the_model_before_the_ledger():
    ledger = _ledger("ledger_op_model.csv")
    op = _family_op("own_def3")
    run_eval.record_test_look({"look": "first", "reason": ""}, run_name="other__test",
                              model_sha="9" * 64, op_sha="8" * 64, bundle_run="other",
                              family="other", schema="def3", threshold=0.3)
    rows_before = ledger.read_text(encoding="utf-8")
    good = _devall("own_def3_devall", "own_def3")
    # another family's model: refused on identity, not as a "second look"
    stranger = _devall("stranger_def3_devall", "stranger_def3")
    _raises(lambda: run_eval.run(stranger, "ultralytics", "def3", "test", operating_point=op,
                                 second_look="would be recorded"),
            EvalGuardError, "does not belong", "family 'own_def3'")
    # same family, but another recipe than the fold models the op was frozen from
    drift = _devall("own_def3_devall_e60", "own_def3", b"short retrain", epochs=60)
    _raises(lambda: run_eval.run(drift, "ultralytics", "def3", "test", operating_point=op,
                                 second_look="would be recorded"),
            EvalGuardError, "epochs differs")
    # the op's fold bundles must be imported to verify that
    orphan = _family_op("own_def3", "op_orphan.json",
                        sources=[{"bundle_run": "never_imported_fold0", "fold": "fold0"}])
    _raises(lambda: run_eval.run(good, "ultralytics", "def3", "test", operating_point=orphan,
                                 second_look="would be recorded"),
            EvalGuardError, "not imported")
    assert ledger.read_text(encoding="utf-8") == rows_before, "a refused test eval must not be recorded"
    # the matching model passes the identity check (and only then meets the ledger)
    _raises(lambda: run_eval.run(good, "ultralytics", "def3", "test", operating_point=op),
            EvalGuardError, "second look")
    _raises(lambda: run_eval.run(good, "ultralytics", "def3", "test", operating_point=op,
                                 second_look="final retrain"), Reached)


def test_zz_results_csv_untouched():
    now = RESULTS.read_bytes() if RESULTS.exists() else None
    assert now == RESULTS_BEFORE, "tests must never append to ml/reports/results_v2.csv"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if failed else 0)
