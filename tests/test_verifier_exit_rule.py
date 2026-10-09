"""
The pre-registered exit rule (ml/verifier/exit_rule.py, PREREGISTRATION.md):
the decision (ship / no-ship / invalid-smoke) and that nothing but the
pre-registered run can reach "ship" — exit-time --set / --config / --threads /
--latency-photos, a config file edited in place, uncommitted code, a
random-init backbone in ANY cross-fit model. Also the exact (b) boundary
(exactly 30 % fewer false alarms passes) and that an earlier exit report is
archived, never overwritten.

run() goes end to end on a synthetic vrun in a temp ML_DATA_ROOT. The heavy
stages (ONNX latency bench, bootstrap, eval-run loading, Verifier) are stubs
whose numbers each test chooses; crossfit_thresholds (b) runs for real.

Run (no pytest needed):  ml/.venv/bin/python -B tests/test_verifier_exit_rule.py
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="zander_verifier_exit_"))
os.environ["ML_DATA_ROOT"] = str(TMP)  # before config.settings is imported
sys.path.insert(0, str(ROOT / "ml"))

from config.settings import settings  # noqa: E402
from verifier import config as vconfig  # noqa: E402
from verifier import exit_rule  # noqa: E402
from verifier.common import EXIT_JSON, EXIT_MD, FOLDS, STACKER, vrun_dir  # noqa: E402
from verifier.config import load_config  # noqa: E402
from verifier.crossfit import make_plan  # noqa: E402

assert settings.ml_data_root == TMP
NAMES = ["Nicht ausreichend Lot", "Brücke", "Fahne"]
PLAN = make_plan()
CLEAN = {"git_sha": "1" * 40, "git_dirty": False}

# ------------------------------------------------------------------ stubs

STATE: dict = {}


def reset(primary=False, ver_ms=300.0, det_ms=1200.0, git=None):
    """Defaults = a NO-SHIP run: slow (verifier 300 ms > 200, total 1.5 s > 1.0),
    primary fails; (b) is decided by the fixture's false-alarm counts."""
    STATE.update(primary=primary, ver_ms=ver_ms, det_ms=det_ms, git=dict(git or CLEAN),
                 lat_args=None)


class _Verifier:
    def __init__(self, *a, **k):
        pass


def _paired_recall(det_photos, casc_photos, c, n_boot=1000, seed=0):
    return {"n_photos": len(det_photos), "n_groups": 1, "n_gt": 5, "recall_detector": 0.5,
            "recall_cascade": 0.6, "diff": 0.1, "bootstrap_n": n_boot, "ci_unit": "board group",
            "diff_ci95": [0.01, 0.2] if STATE["primary"] else [-0.1, 0.2],
            "passed": STATE["primary"]}


def _latency(cand, stacker_path, n_photos, warmup, threads=0, n_candidates=50):
    STATE["lat_args"] = {"n_photos": n_photos, "warmup": warmup, "threads": threads}
    v, d = STATE["ver_ms"], STATE["det_ms"]
    return {"host": "stub", "n_photos": n_photos, "warmup": warmup,
            "threads": threads or "ort-default", "detector_model": "m.onnx",
            "n_candidates": n_candidates, "detector_ms": {"p50": d, "p95": d},
            "verifier_actual_ms": {"p50": v, "p95": v}, "verifier_50_ms": {"p50": v, "p95": v},
            "total_50_ms": {"p50": d + v, "p95": d + v},
            "verifier_50_native_ms": {"p50": v, "p95": v}, "note": "stub"}


exit_rule.Verifier = _Verifier
exit_rule.paired_recall = _paired_recall
exit_rule.latency = _latency
exit_rule.candidate_ceiling = lambda cand, det_by_id, nal: {
    "n_gt": 0, "topk_share": None, "all_detections_share": None, "top_k": 50}
exit_rule.cascade_photos = lambda cand, oof, k: [
    {"capture_id": p["capture_id"]} for p in cand["photos"] if p["fold"] == k]
exit_rule.load_predictions = lambda run: {"photos": list(STATE["det"].values())}
exit_rule.git_state = lambda: dict(STATE["git"])


# ------------------------------------------------------------------ fixtures


def fa_fixture(det_fa: int, casc_fa: int, escape: bool = False):
    """One iO photo in fold0 with `det_fa` NAL detections >= t(k) (all false
    alarms) and as many stacked NAL candidates, `casc_fa` of them >= tau(k).
    `escape`: plus a positive photo the detector flags and the cascade drops."""
    photos, cands, rows, det = [], [], [], {}

    def add(cid, gt, boxes, p_flags):
        photos.append({"capture_id": cid, "fold": "fold0", "gt": gt,
                       "gt_verdict": "NiO" if gt else "iO", "pose": "seated"})
        det[cid] = {"capture_id": cid, "boxes": boxes, "scores": [0.9] * len(boxes),
                    "labels": [0] * len(boxes)}
        for b, flag in zip(boxes, p_flags):
            i = len(cands)
            cands.append({"id": i, "capture_id": cid, "fold": "fold0", "kind": "det", "class": 0,
                          "box": b, "det_score": 0.9})
            rows.append({"id": i, "p_outer": 0.9 if flag else 0.1})

    boxes = [[20.0 * i, 0.0, 20.0 * i + 8.0, 8.0] for i in range(det_fa)]
    add("io_photo", [], boxes, [i < casc_fa for i in range(det_fa)])
    if escape:
        g = [500.0, 500.0, 530.0, 530.0]
        add("nio_photo", [[0, g]], [g], [False])
    oof = {"stack_classes": [0], "rows": rows,
           "folds": {k: {"t_det": 0.5, "tau": 0.5, "hi": {NAMES[0]: None}} for k in FOLDS}}
    return {"candidates": cands, "photos": photos, "class_names": NAMES}, oof, det


def make_vrun(vrun, cfg=None, det_fa=90, casc_fa=80, cand_over=None, oof_over=None,
              train_log="ok"):
    """A complete synthetic vrun as mine + crossfit + export would leave it."""
    cfg = cfg or load_config()
    d = vrun_dir(vrun, create=True)
    fa_cand, fa_oof, det = fa_fixture(det_fa, casc_fa)
    STATE["det"] = det
    cand = {"format_version": 1, "kind": "zander-verifier-candidates", "vrun": vrun,
            "schema": "def3", "class_names": NAMES, "family": "fam_def3",
            "runs": [{"eval_run": f"fam_def3_{k}", "bucket": k, "model_sha256": str(i) * 64}
                     for i, k in enumerate(FOLDS)],
            "smoke": False, "oof_violations": [], "fraction": 1.0,
            "config": cfg.to_dict(), "config_sha256": cfg.sha256, **CLEAN,
            "stack_classes": [0], "passthrough_classes": [1, 2],
            "photos": fa_cand["photos"], "candidates": fa_cand["candidates"]}
    cand.update(cand_over or {})
    data = json.dumps(cand, ensure_ascii=False).encode("utf-8")
    (d / "candidates.json").write_bytes(data)
    oof = {"format_version": 1, "kind": "zander-verifier-oof", "vrun": vrun,
           "candidates_sha256": hashlib.sha256(data).hexdigest(), "smoke": False,
           "fraction": 1.0, "class_names": NAMES, "stack_classes": [0], "t_star": 0.4,
           "operating_point": {}, "plan": PLAN, "folds": fa_oof["folds"],
           "final": {"t_star": 0.4, "tau": 0.5}, "auroc": {}, "weights_source": "imagenet1k_v1",
           "config": cfg.to_dict(), "config_sha256": cfg.sha256, **CLEAN,
           "rows": fa_oof["rows"], "cascade_runs": []}
    oof.update(oof_over or {})
    (d / "oof_scores.json").write_text(json.dumps(oof, ensure_ascii=False), encoding="utf-8")
    if train_log == "ok":
        train_log = {"models": [{"name": n, "weights_source": "imagenet1k_v1"}
                                for n in PLAN["models"]]}
    if train_log is not None:
        (d / "train_log.json").write_text(json.dumps(train_log), encoding="utf-8")
    (d / STACKER).write_text(json.dumps({"kind": "zander-stacker", "exit": None}), encoding="utf-8")
    return d


def run_exit(vrun, **kw):
    """make_vrun kwargs + run() kwargs (cfg, threads, latency_photos)."""
    run_kw = {k: kw.pop(k) for k in ("cfg", "threads", "latency_photos") if k in kw}
    make_vrun(vrun, cfg=kw.pop("vrun_cfg", None), **kw)
    return exit_rule.run(vrun, **run_kw)


def has(report, *needles):
    text = " | ".join(report["invalid_reasons"])
    return all(n in text for n in needles)


# ------------------------------------------------------------------ decision


def test_decision_truth_table():
    from verifier.exit_rule import decide

    assert decide([], True, True, False) == "ship"
    assert decide([], True, False, True) == "ship"
    assert decide([], True, True, True) == "ship"
    assert decide([], False, True, True) == "no-ship"  # the latency gate is required
    assert decide([], True, False, False) == "no-ship"
    assert decide(["smoke"], True, True, True) == "invalid-smoke"


def test_preregistered_run_ships_or_not_on_its_numbers():
    reset()
    r = run_exit("e_noship")
    assert r["decision"] == "no-ship" and r["invalid_reasons"] == [], r["invalid_reasons"]
    reset(primary=True, ver_ms=100.0, det_ms=700.0)
    r = run_exit("e_ship")
    assert r["decision"] == "ship" and r["invalid_reasons"] == []
    assert STATE["lat_args"] == {"n_photos": 20, "warmup": 2, "threads": 0}
    assert r["exit_run"]["config"]["overrides"] == {} and r["exit_run"]["threads"] == 0
    # (b) alone ships: exactly 30 % fewer NAL false alarms (90 -> 63), no added escape
    reset(ver_ms=100.0, det_ms=700.0)
    r = run_exit("e_ship_b", casc_fa=63)
    assert r["secondary_crossfit_thresholds"]["passed"] is True and r["decision"] == "ship"
    sdoc = json.loads((vrun_dir("e_ship_b") / STACKER).read_text(encoding="utf-8"))
    assert sdoc["exit"]["decision"] == "ship" and sdoc["exit"]["reasons"] == []


# ------------------------------------------------------------------ V1: exit-time settings


def test_exit_time_overrides_cannot_turn_no_ship_into_ship():
    """The confirmed exploit: gates loosened at exit time flipped no-ship ->
    ship with no invalid reason (and "ship" unlocks the cascade test bucket)."""
    loose = load_config(overrides={"exit.latency_verifier_p50_ms": 500,
                                   "exit.latency_total_p50_s": 2, "exit.fa_reduction": 0})
    reset()
    r = run_exit("e_loose", cfg=loose)
    assert r["latency"]["gate"]["passed"] and r["secondary_crossfit_thresholds"]["passed"]
    assert r["decision"] == "invalid-smoke", r["decision"]
    assert has(r, "exit:", "overrides", "exit.fa_reduction")
    sdoc = json.loads((vrun_dir("e_loose") / STACKER).read_text(encoding="utf-8"))
    assert sdoc["exit"]["decision"] == "invalid-smoke"
    # another config FILE with identical content is not "the committed file" either
    copy = TMP / "elsewhere" / "mnv3s_2crop.yaml"
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(vconfig.DEFAULT_CONFIG, copy)
    reset(primary=True, ver_ms=100.0, det_ms=700.0)
    r = run_exit("e_other_path", cfg=load_config(copy))
    assert r["decision"] == "invalid-smoke" and has(r, "exit:", "not the pre-registered one")


def test_exit_threads_and_latency_photos_must_be_the_defaults():
    reset(primary=True, ver_ms=100.0, det_ms=700.0)
    r = run_exit("e_threads", threads=2)
    assert r["decision"] == "invalid-smoke" and has(r, "--threads 2")
    assert STATE["lat_args"]["threads"] == 2
    r = run_exit("e_photos", latency_photos=5)
    assert r["decision"] == "invalid-smoke" and has(r, "--latency-photos 5")
    assert STATE["lat_args"]["n_photos"] == 5
    r = run_exit("e_photos20", latency_photos=20)  # the pre-registered count, given explicitly
    assert r["decision"] == "ship" and r["invalid_reasons"] == []


def test_earlier_reports_are_archived_never_overwritten():
    reset()
    first = run_exit("e_archive")
    assert first["decision"] == "no-ship" and first["earlier_reports"] == []
    reset(primary=True, ver_ms=100.0, det_ms=700.0)
    second = exit_rule.run("e_archive")
    third = exit_rule.run("e_archive")  # same second: still a distinct archive folder
    d = vrun_dir("e_archive")
    archived = sorted((d / "exit_archive").iterdir())
    assert len(archived) == 2, archived
    assert all({p.name for p in a.iterdir()} == {EXIT_JSON, EXIT_MD} for a in archived)
    old = json.loads((archived[0] / EXIT_JSON).read_text(encoding="utf-8"))
    assert old["decision"] == "no-ship" and old["created"] == first["created"]
    assert [e["decision"] for e in second["earlier_reports"]] == ["no-ship"]
    assert [e["decision"] for e in third["earlier_reports"]] == ["no-ship", "ship"]
    assert "2 earlier exit report(s)" in (d / EXIT_MD).read_text(encoding="utf-8")
    sdoc = json.loads((d / STACKER).read_text(encoding="utf-8"))
    assert sdoc["exit"]["earlier_decisions"] == ["no-ship", "ship"]


def test_first_valid_decision_binds():
    """A later exit only re-measures latency: re-running until the gate passes
    must not turn the first valid no-ship into a ship (it is still reported)."""
    reset()  # slow verifier: no-ship
    first = run_exit("e_bind")
    assert first["decision"] == first["binding_decision"] == "no-ship"
    assert first["binding_from"] is None
    reset(primary=True, ver_ms=100.0, det_ms=700.0)  # "faster" on the re-run
    second = exit_rule.run("e_bind")
    assert second["decision"] == "ship" and second["binding_decision"] == "no-ship"
    assert second["binding_from"].startswith("exit_archive/")
    d = vrun_dir("e_bind")
    sdoc = json.loads((d / STACKER).read_text(encoding="utf-8"))
    assert sdoc["exit"]["decision"] == "no-ship" and sdoc["exit"]["run_decision"] == "ship"
    assert "Binding decision: NO-SHIP" in (d / EXIT_MD).read_text(encoding="utf-8")
    # an invalid first exit binds nothing: the first VALID one decides
    loose = load_config(overrides={"exit.fa_reduction": 0})
    reset(primary=True, ver_ms=100.0, det_ms=700.0)
    r = run_exit("e_bind2", cfg=loose)
    assert r["decision"] == "invalid-smoke"
    r = exit_rule.run("e_bind2")
    assert r["decision"] == r["binding_decision"] == "ship" and r["binding_from"] is None
    sdoc = json.loads((vrun_dir("e_bind2") / STACKER).read_text(encoding="utf-8"))
    assert sdoc["exit"]["decision"] == "ship"


# ------------------------------------------------------------------ V2: "as committed"


def test_config_sha_is_pinned_to_the_preregistered_file():
    from verifier.config import PREREGISTERED_CONFIG_SHA256

    cfg = load_config()
    assert cfg.sha256 == PREREGISTERED_CONFIG_SHA256 and cfg.is_default
    crlf = TMP / "crlf" / "mnv3s_2crop.yaml"  # a core.autocrlf checkout of the same commit
    crlf.parent.mkdir(parents=True, exist_ok=True)
    crlf.write_bytes(vconfig.DEFAULT_CONFIG.read_bytes().replace(b"\n", b"\r\n"))
    assert load_config(crlf).sha256 == PREREGISTERED_CONFIG_SHA256


def test_config_edited_in_place_is_not_the_preregistered_one():
    """Same path, other content: is_default used to stay true (path check only)."""
    edited = TMP / "edited" / "mnv3s_2crop.yaml"
    edited.parent.mkdir(parents=True, exist_ok=True)
    edited.write_text(vconfig.DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
        "fa_reduction: 0.30", "fa_reduction: 0.0"), encoding="utf-8")
    saved = vconfig.DEFAULT_CONFIG
    vconfig.DEFAULT_CONFIG = edited  # "the committed path" now holds edited content
    try:
        cfg = load_config()
        assert cfg.exit.fa_reduction == 0.0 and not cfg.overrides
        assert not cfg.is_default
        reset()
        r = run_exit("e_edited", cfg=cfg, vrun_cfg=cfg)
        assert r["secondary_crossfit_thresholds"]["passed"]  # the edit loosened (b)...
        assert r["decision"] == "invalid-smoke"  # ...but the run is no longer the pre-registered one
        assert has(r, "mine:", "crossfit:", "exit:", "sha256")
    finally:
        vconfig.DEFAULT_CONFIG = saved


def test_uncommitted_code_is_invalid_at_every_step():
    reset(primary=True, ver_ms=100.0, det_ms=700.0)
    r = run_exit("e_dirty_x", oof_over={"git_dirty": True})
    assert r["decision"] == "invalid-smoke" and has(r, "crossfit:", "uncommitted")
    r = run_exit("e_dirty_m", cand_over={"git_dirty": None})  # mine did not record it
    assert r["decision"] == "invalid-smoke" and has(r, "mine:", "code state unknown")
    reset(primary=True, ver_ms=100.0, det_ms=700.0, git={"git_sha": "2" * 40, "git_dirty": True})
    r = run_exit("e_dirty_e")
    assert r["decision"] == "invalid-smoke" and has(r, "exit:", "uncommitted")
    assert r["git"]["exit"] == {"sha": "2" * 40, "dirty": True}


# ------------------------------------------------------------------ V5: exact (b) boundary


def test_exactly_30_percent_fewer_false_alarms_passes():
    for det_fa, casc_fa, ok in ((90, 63, True), (90, 64, False), (170, 119, True),
                                (180, 126, True), (330, 231, True), (10, 7, True),
                                (10, 8, False), (0, 0, False)):
        cand, oof, det = fa_fixture(det_fa, casc_fa)
        s = exit_rule.crossfit_thresholds(cand, oof, det, 0, 0.30)
        assert (s["pooled"].get("det_fa", 0), s["pooled"].get("casc_fa", 0)) == (det_fa, casc_fa)
        assert s["passed"] is ok, (det_fa, casc_fa, s["passed"])
    cand, oof, det = fa_fixture(90, 10, escape=True)  # any added photo escape fails (b)
    s = exit_rule.crossfit_thresholds(cand, oof, det, 0, 0.30)
    assert s["pooled"]["added_escapes"] == 1 and s["passed"] is False


# ------------------------------------------------------------------ V6: every model's backbone


def test_random_init_in_any_crossfit_model_is_invalid():
    reset(primary=True, ver_ms=100.0, det_ms=700.0)
    log = {"models": [{"name": n, "weights_source": "imagenet1k_v1"} for n in PLAN["models"]]}
    log["models"][1]["weights_source"] = "none (ImageNet weights unavailable: URLError: ...)"
    bad = log["models"][1]["name"]
    r = run_exit("e_rand", train_log=log)  # V_final (oof.weights_source) is fine
    assert r["decision"] == "invalid-smoke" and has(r, "not ImageNet-pretrained", bad)
    r = run_exit("e_nolog", train_log=None)
    assert r["decision"] == "invalid-smoke" and has(r, "train_log.json missing")
    short = {"models": log["models"][2:3]}
    short["models"][0]["weights_source"] = "imagenet1k_v1"
    r = run_exit("e_short", train_log=short)
    assert r["decision"] == "invalid-smoke" and has(r, "lacks the cross-fit models")
    r = run_exit("e_vfinal", oof_over={"weights_source": "none"})
    assert r["decision"] == "invalid-smoke" and has(r, "V_final backbone")


def test_smoke_and_fraction_stay_invalid():
    reset(primary=True, ver_ms=100.0, det_ms=700.0)
    r = run_exit("e_smoke", cand_over={"smoke": True, "oof_violations": ["shared model"]})
    assert r["decision"] == "invalid-smoke" and has(r, "smoke", "shared model")
    r = run_exit("e_frac", cand_over={"fraction": 0.2})
    assert r["decision"] == "invalid-smoke" and has(r, "20% photo sample")


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
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
