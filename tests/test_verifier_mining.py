"""
Out-of-fold candidate mining (ml/verifier/mine.py) and the nested cross-fit
plan (ml/verifier/crossfit.py): label rules, top-k, the OOF guard (synthetic
eval runs in a temp ML_DATA_ROOT, plus the real smoke runs when present), and
the nesting invariants — no fold-k candidate in any model used on fold k, and
stacker rows only from inner out-of-fold scores.

Run (no pytest needed):  ml/.venv/bin/python -B tests/test_verifier_mining.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "ml"
TMP = Path(tempfile.mkdtemp(prefix="zander_verifier_mining_"))
os.environ["ML_DATA_ROOT"] = str(TMP)  # before config.settings is imported
sys.path.insert(0, str(ML))

from config.settings import settings  # noqa: E402
from verifier.config import load_config  # noqa: E402
from verifier.crossfit import check_plan, make_plan, stacker_rows  # noqa: E402
from verifier.mine import (  # noqa: E402
    OOFGuardError, check_oof, family_runs, label_candidate, load_candidates, mine, run_info,
    top_k_indices,
)

assert settings.ml_data_root == TMP
FOLDS = ["fold0", "fold1", "fold2"]
NAMES = ["Nicht ausreichend Lot", "Brücke", "Fahne"]
CFG = load_config(overrides={"candidates.min_pos_per_class": 2, "candidates.top_k": 3})


# ------------------------------------------------------------------ fixtures


def make_bundle(name: str, fold: str, content: bytes) -> Path:
    d = TMP / "run_dirs" / name / "bundle"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.onnx").write_bytes(content)
    (d / "meta.json").write_text(json.dumps({"run_name": name, "fold": fold, "family": "fam"}))
    return d / "model.onnx"


def photo(cid, gt, boxes, scores, labels, other_gt=()):
    return {"capture_id": cid, "group": f"g_{cid}", "board": "A", "pose": "seated",
            "gt_verdict": "NiO" if gt or other_gt else "iO",
            "gt": [[c, list(b)] for c, b in gt], "other_gt": [[n, list(b)] for n, b in other_gt],
            "boxes": [list(b) for b in boxes], "scores": list(scores), "labels": list(labels),
            "width": 2616, "height": 1960}


def make_run(run: str, bucket: str, model: Path, n: int = 0):
    d = settings.eval_dir / run
    d.mkdir(parents=True, exist_ok=True)
    g = (100, 100, 130, 130)
    photos = [
        # positive: an exact hit, a shifted hit (IoU 0.5), an ambiguous one (IoU 0.2)
        photo(f"{bucket}_p{n}", [(0, g)],
              [(100, 100, 130, 130), (110, 100, 140, 130), (120, 100, 150, 130), (900, 900, 930, 930)],
              [0.9, 0.8, 0.7, 0.05], [0, 0, 0, 0]),
        # positive + a Brücke detection sitting on the NAL box (wrong class)
        photo(f"{bucket}_q{n}", [(0, g)], [(100, 100, 130, 130), (100, 100, 130, 130)],
              [0.6, 0.4], [0, 1]),
        # iO photo: background false alarm only
        photo(f"{bucket}_n{n}", [], [(500, 500, 530, 530)], [0.3], [0]),
    ]
    (d / "predictions.json").write_text(json.dumps(
        {"schema": "def3", "class_names": NAMES, "bucket": bucket, "model": str(model),
         "photos": photos}), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(
        {"model": str(model), "adapter": "ultralytics", "threshold": 0.3}), encoding="utf-8")


def valid_family():
    for k in FOLDS:
        make_run(f"fam_{k}", k, make_bundle(f"fam_{k}", k, f"model-{k}".encode()))


def smoke_like():
    m = make_bundle("smk", "fold0", b"one-model")
    for k in FOLDS:
        make_run(f"smk_{k}", k, m)


# ------------------------------------------------------------------ labels


def test_label_rules():
    g = (100, 100, 130, 130)
    gt = [(0, g), (1, (300, 300, 330, 330))]
    other = [("Fahne", (600, 600, 630, 630))]
    assert label_candidate(g, 0, gt, other) == (1, None, 1.0)
    lab, neg, v = label_candidate((110, 100, 140, 130), 0, gt, other)  # IoU 600/1200
    assert (lab, neg) == (1, None) and abs(v - 0.5) < 1e-12
    lab, neg, v = label_candidate((120, 100, 150, 130), 0, gt, other)  # IoU 300/1500
    assert (lab, neg) == (-1, None) and abs(v - 0.2) < 1e-12
    assert label_candidate((126, 100, 156, 130), 0, gt, other)[:2] == (0, "background")
    # a Brücke detection on the NAL box: a real defect, wrong class
    assert label_candidate(g, 1, gt, other)[:2] == (0, "wrong_class")
    # NAL detection half on the Brücke box (IoU 600/1200 = 0.5) -> wrong_class
    assert label_candidate((310, 300, 340, 330), 0, gt, other)[:2] == (0, "wrong_class")
    # NAL detection partly on the Brücke box (IoU 300/1500 = 0.2) -> other_defect
    assert label_candidate((320, 300, 350, 330), 0, gt, other)[:2] == (0, "other_defect")
    # NAL detection on an out-of-schema box (IoU 600/1200) -> other_defect
    assert label_candidate((610, 600, 640, 630), 0, gt, other)[:2] == (0, "other_defect")


def test_top_k_is_stable_on_ties():
    assert top_k_indices([0.5, 0.9, 0.5, 0.1], 3) == [1, 0, 2]
    assert top_k_indices([0.2], 5) == [0] and top_k_indices([], 5) == []


# ------------------------------------------------------------------ OOF guard


def test_guard_accepts_three_distinct_own_fold_models():
    valid_family()
    infos = [run_info(r) for r in family_runs("fam")]
    assert [i["fold"] for i in infos] == FOLDS and [i["fold_source"] for i in infos] == ["meta.json"] * 3
    assert check_oof(infos) == []
    path = mine(family="fam", vrun="t_valid", cfg=CFG)
    doc, sha = load_candidates("t_valid")
    assert path.exists() and len(sha) == 64
    assert doc["smoke"] is False and doc["oof_violations"] == []
    assert doc["stack_classes"] == [0] and doc["passthrough_classes"] == [1, 2]
    det = [c for c in doc["candidates"] if c["kind"] == "det"]
    # top-k 3: the 0.05 detection of the first photo is not a candidate
    assert len(det) == 3 * (3 + 2 + 1)
    assert all(c["det_score"] != 0.05 for c in det)
    by_label = sorted((c["class"], c["label"], c["neg_type"]) for c in det if c["fold"] == "fold0")
    assert by_label == sorted([(0, 1, None), (0, 1, None), (0, -1, None), (0, 1, None),
                               (1, 0, "wrong_class"), (0, 0, "background")])
    gt = [c for c in doc["candidates"] if c["kind"] == "gt"]
    assert len(gt) == 3 * 2 and all(c["label"] == 1 and c["det_score"] is None for c in gt)
    assert [c["id"] for c in doc["candidates"]] == list(range(len(doc["candidates"])))
    assert not any(c["kind"] == "smoke_gt" for c in doc["candidates"])


def test_guard_rejects_shared_non_oof_models_without_the_flag():
    smoke_like()
    runs = family_runs("smk")
    v = check_oof([run_info(r) for r in runs])
    assert any("fold1" in x and "not out-of-fold" in x for x in v)
    assert any("share model files" in x for x in v)
    try:
        mine(runs=runs, vrun="t_smoke", cfg=CFG)
        raise AssertionError("non-OOF runs must be refused without --allow-non-oof")
    except OOFGuardError:
        pass
    mine(runs=runs, vrun="t_smoke", cfg=CFG, allow_non_oof=True)
    doc, _ = load_candidates("t_smoke")
    assert doc["smoke"] is True and doc["oof_violations"] == v
    assert any(c["kind"] == "smoke_gt" for c in doc["candidates"])  # smoke-only injection


def test_guard_hard_errors_even_with_the_flag():
    m = make_bundle("hard", "fold0", b"hard")
    make_run("hard_fold0", "fold0", m)
    make_run("hard_fold1", "fold1", m)
    make_run("hard_test", "test", m)
    for runs in (["hard_fold0", "hard_fold1", "hard_test"], ["hard_fold0", "hard_fold1"]):
        try:
            mine(runs=runs, vrun="t_hard", cfg=CFG, allow_non_oof=True)
            raise AssertionError(f"{runs} must always be refused")
        except OOFGuardError:
            pass


def test_real_smoke_runs_are_refused_without_the_flag():
    """The copied smoke runs (one fold0 model evaluated on all three folds)."""
    real = ROOT / "ml_data" / "eval"
    if not all((real / f"smoke_def3_{k}" / "predictions.json").exists() for k in FOLDS):
        print("  (skip: smoke eval runs not present)")
        return
    code = (
        "import json, sys; sys.path.insert(0, '.');"
        "from verifier.mine import run_info, check_oof;"
        "print(json.dumps(check_oof([run_info(f'smoke_def3_fold{k}') for k in range(3)])))"
    )
    env = {k: v for k, v in os.environ.items() if k != "ML_DATA_ROOT"}
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=ML, env=env,
                         capture_output=True, text=True, check=True)
    violations = json.loads(out.stdout.strip().splitlines()[-1])
    assert len(violations) == 3, violations  # fold1 + fold2 not OOF, one shared sha


# ------------------------------------------------------------------ cross-fit plan


def test_plan_nesting_invariants():
    plan = make_plan()
    check_plan(plan)
    models = plan["models"]
    for k, o in plan["outer"].items():
        used = [o["outer_model"], *o["inner_scorer"].values()]
        assert all(k not in models[m]["train"] for m in used), k
        assert sorted(o["stacker_rows"]) == sorted(f for f in FOLDS if f != k)
        assert sorted(models[o["outer_model"]]["train"]) == sorted(o["stacker_rows"])
    assert models["V_final"]["train"] == FOLDS
    # corrupt the plan: the checker must notice
    bad = make_plan()
    bad["models"]["V_fold0"]["train"] = list(FOLDS)
    try:
        check_plan(bad)
        raise AssertionError("an outer model trained on its own fold must be caught")
    except AssertionError as e:
        assert "trained on it" in str(e)
    bad = make_plan()
    bad["outer"]["fold1"]["inner_scorer"]["fold0"] = "U_fold0"
    try:
        check_plan(bad)
        raise AssertionError("an inner fold scored in-sample must be caught")
    except AssertionError as e:
        assert "trained on it" in str(e)


def test_stacker_rows_only_from_inner_oof_scores():
    plan = make_plan()
    ids = list(range(30))
    folds = [FOLDS[i % 3] for i in ids]
    for k in FOLDS:
        rows = stacker_rows(plan, k, ids, folds)
        assert {r for r, _ in rows} == {i for i in ids if folds[i] != k}
        for r, m in rows:
            train = plan["models"][m]["train"]
            assert folds[r] not in train and k not in train and len(train) == 1, (k, r, m)


def test_smoke_run_artifacts_respect_the_nesting_when_present():
    """If the smoke cross-fit was run, every stored inner score comes from a
    model that saw neither the row's fold nor the outer fold."""
    oof = ROOT / "ml_data" / "verifier" / "smoke_v" / "oof_scores.json"
    if not oof.exists():
        print("  (skip: no smoke cross-fit output)")
        return
    doc = json.loads(oof.read_text(encoding="utf-8"))
    for row in doc["rows"]:
        assert sorted(row["ver_inner"]) == sorted(f for f in FOLDS if f != row["fold"])
        for k in row["ver_inner"]:
            scorer = doc["plan"]["outer"][k]["inner_scorer"][row["fold"]]
            assert row["fold"] not in doc["plan"]["models"][scorer]["train"]
            assert k not in doc["plan"]["models"][scorer]["train"]


# ------------------------------------------------------------------ cascade runs


def _cascade_photos():
    g = (100.0, 100.0, 130.0, 130.0)
    base = {"group": "g1", "board": "A", "pose": "seated", "width": 2616, "height": 1960}
    return [
        {**base, "capture_id": "c_pos", "gt_verdict": "NiO", "gt": [(0, g)], "other_gt": [],
         "boxes": [g, (500.0, 500.0, 530.0, 530.0)], "scores": [0.8123456789, 0.1],
         "labels": [0, 1], "det_scores": [0.4123456789, 0.1]},
        {**base, "capture_id": "c_neg", "gt_verdict": "iO", "gt": [], "other_gt": [],
         "boxes": [(700.0, 700.0, 730.0, 730.0)], "scores": [0.05], "labels": [0],
         "det_scores": [0.3]},
    ]


def _write(name, **kw):
    from verifier.cascade_eval import write_run

    base = {"model": "m.onnx+v", "adapter": "ultralytics", "schema": "def3",
            "mean_ms_per_photo": None, "tiler_version": "v2", "split_id": "split_v2",
            "snapshot_id": "s", "dataset_fingerprint": None, "host": "h"}
    source = {"kind": "cascade", "family": "fam", "verifier_run": "v", "stacker_sha256": "0" * 64}
    return write_run(name, _cascade_photos(), NAMES, "fold0", 0.3, "test", base, source,
                     smoke=False, **kw), source


def test_cascade_runs_through_the_legacy_writer():
    from eval import run_eval

    csv = settings.reports_dir / run_eval.RESULTS_FILE
    before = csv.read_bytes() if csv.exists() else None
    metrics, source = _write("fam+v_fold0", record=False, gallery=False)
    d = settings.eval_dir / "fam+v_fold0"
    doc = json.loads((d / "predictions.json").read_text(encoding="utf-8"))
    assert doc["format_version"] == 2 and doc["source"]["kind"] == "cascade"
    assert doc["photos"][0]["det_scores"] == [0.4123456789, 0.1]  # full precision
    assert doc["photos"][0]["width"] == 2616 and not (d / "gallery.html").exists()
    assert (d / "metrics.json").stat().st_mtime_ns >= (d / "predictions.json").stat().st_mtime_ns
    assert (csv.read_bytes() if csv.exists() else None) == before  # no results row
    back = run_eval.load_predictions("fam+v_fold0")
    assert len(back["photos"]) == 2 and metrics["level1_verdict"]["n_positive"] == 1


def test_cascade_runs_through_a_v2_writer_with_keywords():
    from eval import run_eval

    calls = []

    def write_outputs(out_dir, metrics, photos, *, record=True, gallery=True, source=None):
        calls.append({"record": record, "gallery": gallery, "source": source})
        doc = {"format_version": 2, "schema": metrics["schema"],
               "class_names": metrics["class_names"], "bucket": metrics["bucket"],
               "model": metrics["model"], "source": source,
               "photos": [{**{k: p[k] for k in ("capture_id", "group", "board", "pose",
                                                "gt_verdict", "width", "height")},
                           "gt": [[c, list(b)] for c, b in p["gt"]],
                           "other_gt": [[n, list(b)] for n, b in p["other_gt"]],
                           "boxes": [list(b) for b in p["boxes"]], "scores": p["scores"],
                           "labels": p["labels"], "det_scores": p["det_scores"]}
                          for p in photos]}
        (Path(out_dir) / "predictions.json").write_text(json.dumps(doc), encoding="utf-8")
        (Path(out_dir) / "metrics.json").write_text(json.dumps({"run_name": "x"}), encoding="utf-8")

    run_eval.write_outputs = write_outputs
    try:
        _, source = _write("fam+v_fold1", record=True, gallery=False)
        d = settings.eval_dir / "fam+v_fold1"
        written = (d / "predictions.json").read_bytes()
        assert calls == [{"record": True, "gallery": False, "source": source}]
        from verifier.cascade_eval import ensure_predictions_v2

        assert ensure_predictions_v2(d, _cascade_photos(), source) is False  # left untouched
        assert (d / "predictions.json").read_bytes() == written
        # a smoke run never records a row, whatever the caller asks
        from verifier.cascade_eval import write_run

        write_run("fam+v_fold2", _cascade_photos(), NAMES, "fold2", 0.3, "t",
                  {"model": "m", "adapter": "ultralytics", "schema": "def3",
                   "mean_ms_per_photo": None, "tiler_version": "v2", "split_id": "s",
                   "snapshot_id": "s", "dataset_fingerprint": None, "host": "h"},
                  source, smoke=True, record=True, gallery=True)
        assert calls[-1]["record"] is False
    finally:
        del run_eval.write_outputs


def test_test_bucket_gate():
    """Test bucket: needs the ledger (WP-eval) AND a shipped, non-smoke verifier."""
    from eval import run_eval
    from verifier.cascade_eval import cascade_identity, check_test_bucket
    from verifier.common import Refusal

    known = {"model_sha": cascade_identity("a" * 64, "b" * 64), "op_sha": "c" * 64,
             "second_look": None, "run_name": "r", "schema": "def3"}
    had = hasattr(run_eval, "check_test_ledger")
    saved = getattr(run_eval, "check_test_ledger", None)
    try:
        if had:
            del run_eval.check_test_ledger
        try:
            check_test_bucket("ship", False, **known)
            raise AssertionError("no ledger -> refused")
        except Refusal:
            pass
        seen = []

        def check_test_ledger(model_sha, op_sha, second_look=None):
            seen.append((model_sha, op_sha, second_look))
            return {"look": "first", "reason": ""}

        run_eval.check_test_ledger = check_test_ledger
        for decision, smoke in (("no-ship", False), ("invalid-smoke", True), ("ship", True), (None, False)):
            try:
                check_test_bucket(decision, smoke, **known)
                raise AssertionError(f"{decision}/{smoke} must be refused")
            except Refusal:
                pass
        assert seen == []
        assert check_test_bucket("ship", False, **known) == {"look": "first", "reason": ""}
        assert seen == [(known["model_sha"], "c" * 64, None)]
        assert cascade_identity("a" * 64, "b" * 64) != "a" * 64  # a cascade is its own ledger key
    finally:
        if had:
            run_eval.check_test_ledger = saved
        elif hasattr(run_eval, "check_test_ledger"):
            del run_eval.check_test_ledger


def test_cascade_eval_guards_run_before_anything_is_loaded():
    """cascade-eval: dev folds / train_always / no bucket are refused (stage 2
    was fitted on fold0..2), and the detector passes run_eval's guards — all
    before the verifier or a photo is loaded (no stacker.json exists here)."""
    from dataprep.split import load_split
    from eval import run_eval
    from verifier import cascade_eval
    from verifier.common import Refusal

    split_id = load_split()["split_id"]

    def bundle(name, fold, **over):
        d = settings.bundles_dir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.onnx").write_bytes(f"model-{name}".encode())
        meta = {"bundle_version": 2, "run_name": name, "family": "cfam_def3", "fold": fold,
                "framework": "ultralytics", "adapter": "ultralytics", "schema": "def3",
                "classes": NAMES, "split_id": split_id, **over}
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return d / "model.onnx"

    def refused(model, bucket, *needles, schema="def3"):
        try:
            cascade_eval.run("no_such_vrun", model, schema=schema, bucket=bucket,
                             operating_point=TMP / "op.json")
        except Refusal as e:
            for n in needles:
                assert n in str(e), f"{n!r} not in: {e}"
            return
        raise AssertionError(f"bucket {bucket!r} with {model.parent.name} must be refused")

    devall, fold0 = bundle("cfam_def3_devall", "dev_all"), bundle("cfam_def3_fold0", "fold0")
    for b in ("fold0", "fold1", "fold2"):
        refused(fold0, b, "in-sample", "fold")
    refused(devall, "train_always", "training data")
    refused(devall, None, "--bucket")
    refused(fold0, "test", "dev_all")  # test is the dev_all model's one-shot look
    other = bundle("cfam_def3_devall_s1", "dev_all", split_id="split_v1")
    refused(other, "probe_xprod", "split")
    nal = bundle("cfam_nal1_devall", "dev_all", schema="nal1", classes=NAMES[:1])
    refused(nal, "future", "schema")
    refused(TMP / "missing" / "model.onnx", "test", "not found")
    # the dev_all model passes the detector guards and only then reaches the verifier
    try:
        cascade_eval.run("no_such_vrun", devall, bucket="test", operating_point=TMP / "op.json")
        raise AssertionError("no stacker.json: must fail")
    except Refusal as e:
        raise AssertionError(f"the dev_all model must pass the detector guards: {e}")
    except (FileNotFoundError, ValueError):
        pass
    from verifier.cli import build_parser

    for argv in (["--bucket", "fold0"], []):  # the CLI has no default bucket
        try:
            build_parser().parse_args(["cascade-eval", "--model", "m.onnx", *argv])
            raise AssertionError(f"cascade-eval {argv} must be a usage error")
        except SystemExit as e:
            assert e.code == 2
    csv = settings.reports_dir / run_eval.RESULTS_FILE
    assert not any(settings.eval_dir.glob("cfam*")), "nothing may be written"
    assert csv.exists() is False or "cfam" not in csv.read_text(encoding="utf-8")


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
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
    import shutil

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
