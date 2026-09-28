"""
Paired comparison (ml/eval/compare.py): families pooled over their fold runs,
per-side class resolution, the capture-order check, frozen-operating-point
escape ("n/a" without one), the non-inferiority margin, and the frozen
`compare.run` wrapper.

Hand fixture — two families pooled over 2 folds (4 photos, one group each):
  fold0: P1 NiO, NAL GT G1   P2 iO        fold1: P3 NiO, NAL GT G3 + Brücke GT   P4 iO
  famA (def3): P1 hit 0.9 | P2 FP 0.2 | P3 NAL hit 0.8 + Brücke hit 0.7 | P4 -
  famB (nal1): P1 hit 0.6 | P2 FP 0.95 | P3 -                           | P4 FP 0.3
  NAL recall @ FPPI<=0.5: A = 1.0 (2 hits above every FP), B = 0.5 (G1 at 0.6 with
  1 FP/4 photos; G3 never found) -> diff +0.5.
  Frozen ops t_A = t_B = 0.5; joint positives (NAL photos) P1, P3:
  escape A 0/2, B 1/2 (P3) -> -0.5; verdicts: A right & B wrong on P3 and P2 -> b=2,
  c=0 -> exact McNemar p = 2 * 1/4 = 0.5. A dominates every bootstrap resample,
  so the recall-diff CI lies in [0, 1] and A is non-inferior at margin 0.05.

Run: ml/.venv/bin/python -B tests/test_compare.py
"""
import json
import math
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="zander_compare_"))
os.environ["ML_DATA_ROOT"] = str(TMP)  # BEFORE config.settings is imported
sys.path.insert(0, str(ROOT / "ml"))

from config.settings import settings  # noqa: E402
from eval import compare  # noqa: E402
from eval.photo_scorer import class_records, is_negative, is_positive, pscore  # noqa: E402
from eval.run_eval import EvalGuardError, load_predictions  # noqa: E402
from eval.scorer import (  # noqa: E402
    BOOTSTRAP_N, FPPI_OPERATING, _froc_from_stats, _photo_stats, cluster_resample, recall_at_fppi,
)

DEF3 = ["Nicht ausreichend Lot", "Brücke", "Fahne"]
NAL1 = DEF3[:1]
G1, G3, BR = (100, 100, 130, 130), (200, 200, 230, 230), (400, 400, 430, 430)
EVAL = settings.eval_dir
EVAL.mkdir(parents=True, exist_ok=True)
compare._family_runs = lambda name: [f"{name}_fold0", f"{name}_fold1"]  # 2-fold fixture


def _photo(cid, group, verdict, gt, other, dets):
    return {"capture_id": cid, "group": group, "board": "A", "pose": "seated",
            "gt_verdict": verdict, "width": 1000, "height": 1000,
            "gt": [[c, list(b)] for c, b in gt], "other_gt": [[n, list(b)] for n, b in other],
            "boxes": [list(b) for b, _, _ in dets], "scores": [s for _, s, _ in dets],
            "labels": [c for _, _, c in dets]}


def _gt(schema, cid):
    if cid == "P1":
        return [(0, G1)], []
    if cid == "P3":
        return ([(0, G3), (1, BR)], []) if schema == "def3" else ([(0, G3)], [("Brücke", BR)])
    return [], []


def _write_run(name, bucket, schema, dets_by_photo, order, sha=None, fold=None, family=None):
    photos = []
    for cid in order:
        gt, other = _gt(schema, cid)
        verdict = "NiO" if cid in ("P1", "P3") else "iO"
        photos.append(_photo(cid, f"g{cid}", verdict, gt, other, dets_by_photo.get(cid, [])))
    doc = {"format_version": 2, "schema": schema, "class_names": DEF3 if schema == "def3" else NAL1,
           "bucket": bucket, "model": f"/nowhere/{name}.onnx",
           "source": {"kind": "detector", "bundle_run": name, "family": family or name.rsplit("_", 1)[0],
                      "fold": fold or bucket, "model_sha256": sha or (name.encode().hex() * 8)[:64],
                      "model_resolved": f"/nowhere/{name}.onnx", "adapter": "ultralytics",
                      "conf_floor": 0.001, "verifier_run": None, "stacker_sha256": None},
           "photos": photos}
    d = EVAL / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "predictions.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def _op(schema, family, t):
    (EVAL / f"operating_point_{schema}_{family}.json").write_text(json.dumps(
        {"schema": schema, "class_names": DEF3 if schema == "def3" else NAL1, "threshold": t,
         "runs": [f"{family}_fold0", f"{family}_fold1"], "family": family}), encoding="utf-8")


A = {"P1": [(G1, 0.9, 0)], "P2": [((600, 600, 630, 630), 0.2, 0)],
     "P3": [(G3, 0.8, 0), (BR, 0.7, 1)]}
B = {"P1": [(G1, 0.6, 0)], "P2": [((700, 700, 730, 730), 0.95, 0)],
     "P4": [((800, 800, 830, 830), 0.3, 0)]}
for fam, schema, dets in (("famA", "def3", A), ("famB", "nal1", B), ("famA2", "def3", B)):
    _write_run(f"{fam}_fold0", "fold0", schema, dets, ["P1", "P2"])
    _write_run(f"{fam}_fold1", "fold1", schema, dets, ["P3", "P4"])
_write_run("famC_fold0", "fold0", "def3", A, ["P2", "P1"])  # same photos, other order
_write_run("famC_fold1", "fold1", "def3", A, ["P3", "P4"])
_write_run("famD_fold0", "fold0", "def3", A, ["P1", "P2"], sha="d" * 64)  # one model twice
_write_run("famD_fold1", "fold1", "def3", A, ["P3", "P4"], sha="d" * 64)
_write_run("famE_fold0", "fold0", "def3", A, ["P1", "P2"])
_write_run("famE_fold1", "fold1", "def3", A, ["P3", "P4"], fold="fold0")  # scored off its fold
_op("def3", "famA", 0.5)
_op("nal1", "famB", 0.5)


def _raises(fn, exc, *needles):
    try:
        fn()
    except exc as e:
        for n in needles:
            assert n in str(e), f"{n!r} not in: {e}"
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_hand_fixture_pooled_over_two_folds():
    r = compare.compare("famA", "famB", cls="NAL", margin=0.05)
    assert r["kind"] == "family vs family" and r["bucket"] == "oof(fold0+fold1)"
    assert r["runs_a"] == ["famA_fold0", "famA_fold1"] and r["n_photos"] == 4
    assert r["class"] == "Nicht ausreichend Lot" and (r["class_index_a"], r["class_index_b"]) == (0, 0)
    assert (r["recall_a"], r["recall_b"], r["recall_at_0.5fppi_diff"]) == (1.0, 0.5, 0.5), r
    assert r["n_gt"] == 2 and r["n_joint_positive"] == 2
    assert (r["threshold_a"], r["threshold_b"]) == (0.5, 0.5)
    assert r["operating_point_a"] == "operating_point_def3_famA.json"
    assert (r["escape_a"], r["escape_b"], r["escape_diff"]) == (0.0, 0.5, -0.5), r
    assert r["verdict_discordant"] == {"a_right_b_wrong": 2, "b_right_a_wrong": 0}
    assert r["mcnemar_p"] == 0.5
    lo, hi = r["recall_diff_ci95"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0, r["recall_diff_ci95"]
    assert r["g1_decision"] == ("A better" if lo > 0 else
                                "no significant difference -> prefer the faster/simpler model")
    assert r["non_inferior"] is True and "non-inferior" in r["non_inferiority"]
    out = EVAL / "compare_famA__famB__NAL.json"
    assert json.loads(out.read_text(encoding="utf-8"))["recall_a"] == 1.0


def test_reverse_direction_is_not_non_inferior():
    r = compare.compare("famB", "famA", cls="NAL", margin=0.05)
    assert r["recall_at_0.5fppi_diff"] == -0.5 and r["recall_diff_ci95"][1] <= 0.0
    assert r["non_inferior"] is False and "NOT shown" in r["non_inferiority"]


def test_class_resolution_per_side():
    rc = compare.resolve_class
    assert rc("NAL", DEF3) == (0, DEF3[0]) and rc("nal", NAL1) == (0, DEF3[0])
    assert rc("Brücke", DEF3) == (1, "Brücke") and rc("bruecke", DEF3) == (1, "Brücke")
    assert rc("Fahne", DEF3) == (2, "Fahne") and rc("2", DEF3) == (2, "Fahne") and rc(1, DEF3)[0] == 1
    assert rc("any", DEF3) == (None, "any")
    _raises(lambda: rc("xyz", DEF3), ValueError, "not in")
    _raises(lambda: rc(3, DEF3), ValueError, "out of range")
    _raises(lambda: compare.compare("famA", "famB", cls="Brücke", write=False), ValueError, "not in")
    _raises(lambda: compare.compare("famA", "famB", cls="any", write=False), ValueError, "identical")
    same = compare.compare("famA", "famA2", cls="any", write=False)  # same schema: any works
    assert same["class"] == "any" and same["class_index_a"] is None
    br = compare.compare("famA", "famA2", cls="Brücke", write=False)
    assert br["class"] == "Brücke" and br["recall_a"] == 1.0 and br["recall_b"] == 0.0
    assert compare.compare("famA", "famB", cls=0, write=False)["class"] == DEF3[0]


def test_capture_order_is_checked():
    _raises(lambda: compare.compare("famA", "famC", write=False), ValueError, "same order")
    _raises(lambda: compare.compare("famA", "famA_fold0", write=False), ValueError, "same bucket")


def test_family_pooling_refuses_non_oof_runs():
    _raises(lambda: compare.resolve("famD"), EvalGuardError, "same model")
    _raises(lambda: compare.resolve("famE"), EvalGuardError, "validates on fold0")
    _raises(lambda: compare.resolve("nothing_here"), compare.CompareError, "neither")


def test_missing_operating_point_is_na():
    (EVAL / "operating_point_nal1_famB.json").rename(EVAL / "moved_op.json")
    try:
        r = compare.compare("famA", "famB", cls="NAL", write=False)
    finally:
        (EVAL / "moved_op.json").rename(EVAL / "operating_point_nal1_famB.json")
    assert r["operating_point_b"] == "n/a" and r["threshold_b"] is None
    assert r["escape_diff"] is None and r["escape_b"] is None and r["mcnemar_p"] is None
    assert r["recall_at_0.5fppi_diff"] == 0.5  # the defect level never needs a threshold
    assert "n/a" in compare.summary_line(r)


def test_single_runs_and_the_frozen_wrapper():
    r = compare.compare("famA_fold1", "famB_fold1", cls="NAL", write=False)
    assert r["kind"] == "run vs run" and r["bucket"] == "fold1" and r["threshold_a"] == 0.5
    w = compare.run("famA", "famB", class_index=0, seed=0)
    full = compare.compare("famA", "famB", cls="NAL", seed=0)
    assert {k: v for k, v in w.items() if k != "file"} == {k: v for k, v in full.items() if k != "file"}


# ------------------------------------------------ regression vs the old code


def _old_compare(da, db, ta, tb, class_index=0, seed=0):
    """The pre-2026-09-28 eval/compare.run algorithm, verbatim except that the
    two docs and thresholds are passed in instead of read from ml_data/eval."""
    ids_a = [p["capture_id"] for p in da["photos"]]
    if ids_a != [p["capture_id"] for p in db["photos"]] or da["bucket"] != db["bucket"]:
        raise ValueError("runs must cover the same bucket and photos in the same order")
    if da["class_names"][class_index] != db["class_names"][class_index]:
        raise ValueError("class index refers to different classes in the two runs")
    photos_a, photos_b = da["photos"], db["photos"]
    clusters = [p["group"] for p in photos_a]
    stats_a = _photo_stats(class_records(photos_a, class_index), 0.3)
    stats_b = _photo_stats(class_records(photos_b, class_index), 0.3)

    def recall(stats, idx):
        pts, n_gt = _froc_from_stats([stats[i] for i in idx])
        return recall_at_fppi(pts, FPPI_OPERATING)[0] if n_gt else None

    def escape(photos, t, idx):
        pos = [photos[i] for i in idx if is_positive(photos[i])]
        return sum(1 for p in pos if pscore(p) < t) / len(pos) if pos and t is not None else None

    full = list(range(len(photos_a)))
    point = {
        "recall_diff": recall(stats_a, full) - recall(stats_b, full),
        "escape_diff": (escape(photos_a, ta, full) - escape(photos_b, tb, full)
                        if ta is not None and tb is not None else None),
    }
    rng = random.Random(seed)
    boots = {"recall_diff": [], "escape_diff": []}
    for _ in range(BOOTSTRAP_N):
        idx = cluster_resample(len(photos_a), clusters, rng)
        ra, rb = recall(stats_a, idx), recall(stats_b, idx)
        if ra is not None and rb is not None:
            boots["recall_diff"].append(ra - rb)
        ea, eb = escape(photos_a, ta, idx), escape(photos_b, tb, idx)
        if ea is not None and eb is not None:
            boots["escape_diff"].append(ea - eb)

    def ci(xs):
        if not xs:
            return None
        xs = sorted(xs)
        return [round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs)) - 1], 4)]

    b = c = 0
    for pa, pb in zip(photos_a, photos_b):
        if not (is_positive(pa) or is_negative(pa)) or ta is None or tb is None:
            continue
        truth = is_positive(pa)
        ok_a = (pscore(pa) >= ta) == truth
        ok_b = (pscore(pb) >= tb) == truth
        b += ok_a and not ok_b
        c += ok_b and not ok_a
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    result = {
        "run_a": "A", "run_b": "B", "bucket": da["bucket"],
        "class": da["class_names"][class_index],
        "recall_at_0.5fppi_diff": round(point["recall_diff"], 4),
        "recall_diff_ci95": ci(boots["recall_diff"]),
        "escape_diff": None if point["escape_diff"] is None else round(point["escape_diff"], 4),
        "escape_diff_ci95": ci(boots["escape_diff"]),
        "verdict_discordant": {"a_right_b_wrong": b, "b_right_a_wrong": c},
        "mcnemar_p": round(p, 4),
        "ci_unit": "board group",
    }
    rc = result["recall_diff_ci95"]
    decisive = rc is not None and (rc[0] > 0 or rc[1] < 0)
    result["g1_decision"] = (("A better" if rc[0] > 0 else "B better") if decisive
                             else "no significant difference -> prefer the faster/simpler model")
    return result


OLD_KEYS = ["bucket", "class", "recall_at_0.5fppi_diff", "recall_diff_ci95", "escape_diff",
            "escape_diff_ci95", "verdict_discordant", "mcnemar_p", "ci_unit", "g1_decision"]
SMOKE = ROOT / "ml_data" / "eval"
# compare.run("smoke_def3_fold0", "smoke_nal1_fold0") with the pre-change code:
OLD_SMOKE = {"bucket": "fold0", "class": "Nicht ausreichend Lot", "recall_at_0.5fppi_diff": 0.0,
             "recall_diff_ci95": [0.0, 0.0], "ci_unit": "board group",
             "g1_decision": "no significant difference -> prefer the faster/simpler model"}


def _copy_smoke(*names) -> bool:
    if not all((SMOKE / n / "predictions.json").is_file() for n in names):
        print("SKIP smoke regression (ml_data/eval smoke runs absent)")
        return False
    for n in names:
        (EVAL / n).mkdir(parents=True, exist_ok=True)
        for f in ("predictions.json", "metrics.json"):
            shutil.copyfile(SMOKE / n / f, EVAL / n / f)
    legacy = SMOKE / "operating_point_def3_smoke_def3_fold0.json"
    if legacy.is_file():
        shutil.copyfile(legacy, EVAL / legacy.name)
    return True


def test_same_schema_matches_the_old_algorithm_exactly():
    if not _copy_smoke("smoke_def3_fold0"):
        return
    base = json.loads((EVAL / "smoke_def3_fold0" / "predictions.json").read_text(encoding="utf-8"))
    for p in base["photos"]:  # a different model: every 3rd box dropped, scores recalibrated
        keep = [i for i in range(len(p["scores"])) if i % 3 != 1]
        p["boxes"] = [p["boxes"][i] for i in keep]
        p["scores"] = [round(p["scores"][i] ** 0.8, 5) for i in keep]
        p["labels"] = [p["labels"][i] for i in keep]
    (EVAL / "smoke_def3_perturbed").mkdir(exist_ok=True)
    (EVAL / "smoke_def3_perturbed" / "predictions.json").write_text(json.dumps(base), encoding="utf-8")
    da, db = load_predictions("smoke_def3_fold0"), load_predictions("smoke_def3_perturbed")
    for ta, tb in ((0.0056382715702056885, 0.02), (0.01, 0.005)):
        old = _old_compare(da, db, ta, tb)
        new = compare.compare("smoke_def3_fold0", "smoke_def3_perturbed", cls=0,
                              thresholds=(ta, tb), write=False)
        for k in OLD_KEYS:
            assert new[k] == old[k], (k, new[k], old[k])
    assert old["verdict_discordant"] != {"a_right_b_wrong": 0, "b_right_a_wrong": 0}, "weak fixture"


def test_wrapper_reproduces_the_smoke_comparison():
    if not _copy_smoke("smoke_def3_fold0", "smoke_nal1_fold0"):
        return
    r = compare.run("smoke_def3_fold0", "smoke_nal1_fold0")
    for k, v in OLD_SMOKE.items():
        assert r[k] == v, (k, r[k], v)
    # escape now uses frozen operating points: def3 has the legacy smoke op, nal1 none
    assert r["operating_point_a"] == "operating_point_def3_smoke_def3_fold0.json"
    assert r["threshold_a"] == 0.00555 and r["operating_point_b"] == "n/a"
    assert r["escape_diff"] is None and r["mcnemar_p"] is None
    assert (EVAL / "compare_smoke_def3_fold0__smoke_nal1_fold0__NAL.json").is_file()


def test_summary_ranks_runs_g1_and_the_nal_gate():
    from eval import summary

    for fam, schema, dets in (("sumA_def3", "def3", A), ("sumB_def3", "def3", B),
                              ("sumA_nal1", "nal1", B)):
        for k, order in enumerate((["P1", "P2"], ["P3", "P4"])):
            run = f"{fam}_fold{k}"
            _write_run(run, f"fold{k}", schema, dets, order, family=fam)
            (EVAL / run / "metrics.json").write_text(json.dumps(
                {"bucket": f"fold{k}", "n_photos": 2, "mean_ms_per_photo": 400.0 + k,
                 "level1_verdict": {}, "level3_defects": {"per_class": {}}}), encoding="utf-8")
    _op("def3", "sumA_def3", 0.5)
    summary.load_split = lambda: {"split_id": "split_v2", "folds": {"0": [], "1": []},
                                  "probe_boxes": []}  # the 2-fold fixture
    md, js = TMP / "model_selection.md", TMP / "summary.json"
    doc = summary.run(families=["sumA_def3", "sumB_def3", "sumA_nal1"], out_md=md, out_json=js)
    assert doc["ranking"] == ["sumA_def3", "sumB_def3"], doc["ranking"]
    by = {e["family"]: e for e in doc["families"]}
    assert by["sumA_def3"]["nal"]["recall"] == 1.0 and by["sumB_def3"]["nal"]["recall"] == 0.5
    assert by["sumA_def3"]["self_tuned"] is False and by["sumB_def3"]["self_tuned"] is True
    assert by["sumA_def3"]["threshold"] == 0.5 and by["sumA_def3"]["ms_per_photo"] == 400.5
    assert by["sumA_def3"]["escape"]["n"] == 2 and by["sumA_def3"]["arch"] == "sumA"
    g1 = doc["g1"]
    assert (g1["run_a"], g1["run_b"], g1["recall_at_0.5fppi_diff"]) == ("sumA_def3", "sumB_def3", 0.5)
    gate = {g["run_a"]: g for g in doc["nal_gate"]}
    assert gate["sumA_def3"]["run_b"] == "sumA_nal1" and gate["sumA_def3"]["non_inferior"] is True
    assert gate["sumB_def3"]["missing"] is True
    text = md.read_text(encoding="utf-8")
    assert "## G1" in text and "`sumA_def3`" in text and "non-inferior" in text and "self-tuned ⚠" in text
    assert json.loads(js.read_text(encoding="utf-8"))["ranking"] == doc["ranking"]
    # families that cannot be pooled are listed as excluded, not ranked
    for run in ("famA_fold0", "famA_fold1", "famD_fold0", "famD_fold1"):
        (EVAL / run / "metrics.json").write_text(json.dumps(
            {"bucket": run[-5:], "n_photos": 2, "mean_ms_per_photo": 300.0}), encoding="utf-8")
    doc = summary.run(families=["famA", "famD"], out_md=md, out_json=js)
    assert [x["family"] for x in doc["excluded"]] == ["famD"] and doc["ranking"] == ["famA"]


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
