"""
The resumable training matrix (ml/train/matrix.py) with an injected fake
launcher: ok / fail / skip / resume paths, matrix_log.jsonl + summary contents,
exit codes, the bundle sha check, GPU profiles and the output tee — CPU only.

Run: ml/.venv/bin/python -B tests/test_train_matrix.py
"""
import atexit
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "ml"
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="zander_test_train_matrix_"))
atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)
os.environ["ML_DATA_ROOT"] = str(_TMP_ROOT / "ml_data")  # before any settings import
sys.path.insert(0, str(ML))

from config.settings import settings  # noqa: E402
from train import bundle as B  # noqa: E402
from train import dispatch  # noqa: E402
from train import matrix as M  # noqa: E402

# docs/CONTRACTS.md §8.5 + the WP-train brief: every matrix_log.jsonl line has these.
JSONL_FIELDS = {"run_name", "config", "status", "attempt", "resumed", "started", "ended",
                "minutes", "epochs_completed", "gpu", "git_sha", "returncode",
                "bundle_sha256", "error_tail"}
RUN_ROOT = settings.run_dirs_dir


def _fresh(*names, extra_lines=""):
    """A new drive dir + one config per name; the shared run_dirs are wiped."""
    shutil.rmtree(RUN_ROOT, ignore_errors=True)
    RUN_ROOT.mkdir(parents=True)
    base = Path(tempfile.mkdtemp(dir=_TMP_ROOT))
    cfg_dir = base / "configs"
    cfg_dir.mkdir()
    paths = []
    for n in names:
        schema = n.split("_")[1]
        fold = n.rsplit("_", 1)[1]
        p = cfg_dir / f"{n}.yaml"
        p.write_text(f"name: {n}\nmodel: yolo26n.pt\nadapter: ultralytics\ndataset_id: ds_v2_{schema}\n"
                     f"fold: {fold}\nepochs: 2\nimgsz: 640\nbatch: 8\nseed: 1\n{extra_lines}")
        paths.append(str(p))
    return base / "drive", paths


class FakeLauncher:
    """Stands in for `python train/dispatch.py` — plays a scripted action per attempt."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def __call__(self, req):
        self.calls.append((req.item.name, req.attempt))
        assert "--mirror-dir" in req.cmd and str(req.mirror_dir) in req.cmd
        actions = self.script[req.item.name]
        action = actions[min(req.attempt, len(actions)) - 1]
        name = req.item.name
        run_dir, mirror = req.run_root / name, req.mirror_dir / name
        resumed = (mirror / "last.pt").is_file()
        side = {"format": 1, "run_name": name, "config_sha256": req.item.config_sha, "attempts": []}
        if (mirror / "zander_run.json").is_file():
            side = json.loads((mirror / "zander_run.json").read_text())
        start = max([a["end_epoch"] for a in side["attempts"]] or [0])
        if action == "partial":  # one epoch trained + mirrored, then the VM vanished
            side["attempts"].append({"started": B.utc_now(), "start_epoch": start, "end_epoch": start + 1,
                                     "resumed": resumed, "gpu": "fake", "seconds": 60.0})
            for d in (run_dir, mirror):
                d.mkdir(parents=True, exist_ok=True)
                (d / "zander_run.json").write_text(json.dumps(side))
            (mirror / "last.pt").write_bytes(b"ckpt")
            shutil.rmtree(run_dir)  # a new VM: nothing local survives
            return M.LaunchResult(75, "epoch 1/2 done\nsimulated disconnect")
        if action == "ok":
            side["attempts"].append({"started": B.utc_now(), "start_epoch": start, "end_epoch": 2,
                                     "resumed": resumed, "gpu": "fake", "seconds": 120.0})
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "zander_run.json").write_text(json.dumps(side))
            meta = {"run_name": name, "bundle_version": 2, "config_sha256": req.item.config_sha,
                    "dataset_fingerprint": req.item.fingerprint,
                    "epochs_completed": 2, "attempts": side["attempts"]}
            zip_path, _ = B.write_bundle(req.run_root / f"{name}_bundle.zip", {
                "config.yaml": "name: x\n", "meta.json": json.dumps(meta), "metrics.json": "{}",
                "model.onnx": b"onnx", "last.pt": b"weights"})
            return M.LaunchResult(0, f"training...\n{dispatch.BUNDLE_MARKER}{zip_path}")
        if action == "fail":
            return M.LaunchResult(1, "Traceback (most recent call last):\nRuntimeError: CUDA out of memory")
        if action == "config":
            return M.LaunchResult(2, "configuration/data error (not retryable): tiles missing")
        if action == "boom":  # the driver itself fails around the run (a bug, Drive gone, ...)
            raise OSError(107, "Transport endpoint is not connected")
        raise AssertionError(action)


def _run(argv, launcher):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = M.main(argv, launcher=launcher)
    return rc, out.getvalue() + err.getvalue()


def _log(drive):
    return M.read_jsonl(Path(drive) / "matrix_log.jsonl")


def _argv(drive, configs, *extra):
    return ["--configs", *configs, "--drive", str(drive), "--device", "cpu", "--profile", "none", *extra]


# -------------------------------------------------------------------- flows


def test_ok_run_is_published_and_logged():
    drive, cfgs = _fresh("yolo26n_def3_fold0")
    fake = FakeLauncher({"yolo26n_def3_fold0": ["ok"]})
    rc, out = _run(_argv(drive, cfgs), fake)
    assert rc == 0, out
    assert fake.calls == [("yolo26n_def3_fold0", 1)]
    zip_path = drive / "bundles" / "yolo26n_def3_fold0_bundle.zip"
    check = B.verify_bundle(zip_path)
    assert check.ok, check.reason
    assert B.read_sha256_file(B.sha_path_for(zip_path)) == check.sha256
    assert not (drive / "mirror" / "yolo26n_def3_fold0").exists()
    log = _log(drive)
    assert [r["status"] for r in log] == ["started", "ok"]
    for r in log:
        assert JSONL_FIELDS <= set(r), JSONL_FIELDS - set(r)
    ok = log[-1]
    assert ok["bundle_sha256"] == check.sha256 and ok["returncode"] == 0 and ok["attempt"] == 1
    assert ok["resumed"] is False and ok["epochs_completed"] == 2 and ok["epochs_this_attempt"] == 2
    assert ok["gpu"] and ok["git_sha"] and ok["error_tail"] is None
    assert "yolo26n_def3_fold0" in out and " ok " in out  # summary table


def test_failure_never_stops_the_loop_and_exits_1():
    drive, cfgs = _fresh("yolo26n_def3_fold0", "yolo26n_def3_fold1")
    fake = FakeLauncher({"yolo26n_def3_fold0": ["fail", "fail"], "yolo26n_def3_fold1": ["ok"]})
    rc, out = _run(_argv(drive, cfgs, "--max-attempts", "2"), fake)
    assert rc == 1, out
    assert fake.calls == [("yolo26n_def3_fold0", 1), ("yolo26n_def3_fold0", 2), ("yolo26n_def3_fold1", 1)]
    statuses = [(r["run_name"][-5:], r["status"], r["attempt"]) for r in _log(drive)]
    assert statuses == [("fold0", "started", 1), ("fold0", "failed", 1), ("fold0", "started", 2),
                        ("fold0", "failed", 2), ("fold1", "started", 1), ("fold1", "ok", 1)], statuses
    failed = [r for r in _log(drive) if r["status"] == "failed"]
    assert all("CUDA out of memory" in r["error_tail"] and r["returncode"] == 1 for r in failed)
    assert "failed" in out and "1 run(s) failed" in out
    assert B.verify_bundle(drive / "bundles" / "yolo26n_def3_fold1_bundle.zip").ok


def test_finished_bundle_is_skipped_never_retrained():
    drive, cfgs = _fresh("yolo26n_def3_fold0")
    _run(_argv(drive, cfgs), FakeLauncher({"yolo26n_def3_fold0": ["ok"]}))
    sha_before = B.sha256_file(drive / "bundles" / "yolo26n_def3_fold0_bundle.zip")
    (drive / "mirror" / "yolo26n_def3_fold0").mkdir(parents=True)  # leftover of a killed cleanup
    fake = FakeLauncher({"yolo26n_def3_fold0": ["ok"]})
    rc, out = _run(_argv(drive, cfgs), fake)
    assert rc == 0 and fake.calls == [], out
    assert _log(drive)[-1]["status"] == "skipped" and _log(drive)[-1]["warning"] is None
    assert not (drive / "mirror" / "yolo26n_def3_fold0").exists()
    # a changed effective config is WARNED about, but the finished result is kept
    fake = FakeLauncher({"yolo26n_def3_fold0": ["ok"]})
    rc, out = _run(_argv(drive, cfgs, "--set", "epochs=60"), fake)
    assert rc == 0 and fake.calls == [], out
    last = _log(drive)[-1]
    assert last["status"] == "skipped" and "config_sha256" in last["warning"]
    assert "WARNING" in out
    assert B.sha256_file(drive / "bundles" / "yolo26n_def3_fold0_bundle.zip") == sha_before


def test_skip_warns_when_the_tiles_changed_since_the_bundle():
    # T2: a finished bundle trained on other tiles is kept (never overwritten) but WARNED about
    prov = settings.tiles_dir / "ds_v2_def3" / "provenance.json"
    prov.parent.mkdir(parents=True, exist_ok=True)
    try:
        prov.write_text(json.dumps({"dataset_id": "ds_v2_def3", "dataset_fingerprint": "1" * 64}))
        drive, cfgs = _fresh("yolo26n_def3_fold0")
        items, _ = M.load_items(cfgs, {})
        assert items[0].fingerprint == "1" * 64
        assert _run(_argv(drive, cfgs), FakeLauncher({"yolo26n_def3_fold0": ["ok"]}))[0] == 0
        rc, out = _run(_argv(drive, cfgs), FakeLauncher({"yolo26n_def3_fold0": ["ok"]}))
        assert rc == 0 and _log(drive)[-1]["warning"] is None, out
        prov.write_text(json.dumps({"dataset_id": "ds_v2_def3", "dataset_fingerprint": "2" * 64}))
        fake = FakeLauncher({"yolo26n_def3_fold0": ["ok"]})
        rc, out = _run(_argv(drive, cfgs), fake)
        assert rc == 0 and fake.calls == [], out
        last = _log(drive)[-1]
        assert last["status"] == "skipped" and "dataset_fingerprint" in last["warning"], last
        assert "WARNING" in out
    finally:
        prov.unlink()
    assert M.tiles_fingerprint("ds_v2_def3") is None  # tiles not built: no check, no crash


def test_retry_resumes_after_a_disconnect():
    drive, cfgs = _fresh("yolo26n_def3_fold0")
    fake = FakeLauncher({"yolo26n_def3_fold0": ["partial", "ok"]})
    rc, out = _run(_argv(drive, cfgs, "--max-attempts", "2"), fake)
    assert rc == 0, out
    log = _log(drive)
    assert [(r["status"], r["attempt"], r["resumed"]) for r in log] == [
        ("started", 1, False), ("failed", 1, False), ("started", 2, True), ("ok", 2, True)], log
    assert log[1]["returncode"] == 75 and log[1]["epochs_completed"] == 1
    assert log[-1]["epochs_this_attempt"] == 1 and log[-1]["epochs_completed"] == 2


def test_mirror_is_kept_until_a_later_session_verified_the_bundle():
    # T5: Drive uploads in the background — deleting the mirror seconds after
    # writing the bundle could leave neither copy if the VM dies now
    drive, cfgs = _fresh("yolo26n_def3_fold0")
    rc, out = _run(_argv(drive, cfgs, "--max-attempts", "2"),
                   FakeLauncher({"yolo26n_def3_fold0": ["partial", "ok"]}))
    assert rc == 0, out
    mirror = drive / "mirror" / "yolo26n_def3_fold0"
    assert (mirror / "last.pt").is_file() and (mirror / "zander_run.json").is_file(), "mirror kept"
    # the next session reads the bundle back from Drive (sha verified) — only then the mirror goes
    fake = FakeLauncher({"yolo26n_def3_fold0": ["ok"]})
    rc, out = _run(_argv(drive, cfgs), fake)
    assert rc == 0 and fake.calls == [] and not mirror.exists(), out


def test_next_session_resumes_from_the_drive_mirror():
    drive, cfgs = _fresh("yolo26n_def3_fold0")
    rc, _ = _run(_argv(drive, cfgs, "--max-attempts", "1"), FakeLauncher({"yolo26n_def3_fold0": ["partial"]}))
    assert rc == 1
    assert (drive / "mirror" / "yolo26n_def3_fold0" / "last.pt").is_file()
    rc, out = _run(_argv(drive, cfgs), FakeLauncher({"yolo26n_def3_fold0": ["ok"]}))
    assert rc == 0, out
    assert _log(drive)[-1]["status"] == "ok" and _log(drive)[-1]["resumed"] is True
    assert "resume" in out


def test_invalid_bundle_is_moved_aside_and_redone():
    drive, cfgs = _fresh("yolo26n_def3_fold0")
    bundles = drive / "bundles"
    bundles.mkdir(parents=True)
    (bundles / "yolo26n_def3_fold0_bundle.zip").write_bytes(b"half a zip")  # copy died mid-way
    (bundles / "yolo26n_def3_fold0_bundle.zip.sha256").write_text("0" * 64 + "  yolo26n_def3_fold0_bundle.zip\n")
    fake = FakeLauncher({"yolo26n_def3_fold0": ["ok"]})
    rc, out = _run(_argv(drive, cfgs), fake)
    assert rc == 0 and fake.calls == [("yolo26n_def3_fold0", 1)], out
    aside = sorted(p.name for p in bundles.iterdir() if ".invalid-" in p.name)
    assert len(aside) == 2 and all(n.startswith("yolo26n_def3_fold0_bundle.zip") for n in aside), aside
    assert B.verify_bundle(bundles / "yolo26n_def3_fold0_bundle.zip").ok


def test_zip_without_marker_is_not_a_result():
    drive, cfgs = _fresh("yolo26n_def3_fold0")
    (drive / "bundles").mkdir(parents=True)
    (drive / "bundles" / "yolo26n_def3_fold0_bundle.zip").write_bytes(b"x")
    fake = FakeLauncher({"yolo26n_def3_fold0": ["ok"]})
    rc, _ = _run(_argv(drive, cfgs), fake)
    assert rc == 0 and len(fake.calls) == 1


def test_a_driver_error_fails_the_run_never_the_matrix():
    # T1: anything but Ctrl-C raised around one run is a failed attempt; the loop goes on
    drive, cfgs = _fresh("yolo26n_def3_fold0", "yolo26n_def3_fold1")
    fake = FakeLauncher({"yolo26n_def3_fold0": ["boom", "boom"], "yolo26n_def3_fold1": ["ok"]})
    rc, out = _run(_argv(drive, cfgs, "--max-attempts", "2"), fake)
    assert rc == 1, out
    assert fake.calls == [("yolo26n_def3_fold0", 1), ("yolo26n_def3_fold0", 2), ("yolo26n_def3_fold1", 1)]
    failed = [r for r in _log(drive) if r["status"] == "failed"]
    assert len(failed) == 2 and all(r["returncode"] is None and "matrix driver error" in r["error_tail"]
                                    and "Transport endpoint" in r["error_tail"] for r in failed), failed
    assert "driver error" in out and "1 run(s) failed" in out
    assert B.verify_bundle(drive / "bundles" / "yolo26n_def3_fold1_bundle.zip").ok


def test_an_unwritable_matrix_log_never_stops_training():
    # T1: matrix_log.jsonl is bookkeeping — a Drive error there is warned about once
    drive, cfgs = _fresh("yolo26n_def3_fold0", "yolo26n_def3_fold1")
    (drive / "matrix_log.jsonl").mkdir(parents=True)  # every append now raises an OSError
    fake = FakeLauncher({"yolo26n_def3_fold0": ["ok"], "yolo26n_def3_fold1": ["ok"]})
    rc, out = _run(_argv(drive, cfgs), fake)
    assert rc == 0 and len(fake.calls) == 2, out
    assert out.count("cannot append") == 1, out
    for n in ("yolo26n_def3_fold0", "yolo26n_def3_fold1"):
        assert B.verify_bundle(drive / "bundles" / f"{n}_bundle.zip").ok


def test_an_unreadable_bundle_is_never_moved_aside_or_retrained():
    # T1: an I/O error while CHECKING a finished bundle says nothing about the
    # bundle — it must not be moved aside as invalid, and not be trained again
    drive, cfgs = _fresh("yolo26n_def3_fold0", "yolo26n_def3_fold1")
    assert _run(_argv(drive, cfgs[:1]), FakeLauncher({"yolo26n_def3_fold0": ["ok"]}))[0] == 0
    bundles = drive / "bundles"
    zip0 = bundles / "yolo26n_def3_fold0_bundle.zip"
    sha_before = B.sha256_file(zip0)
    real_sha = B.sha256_file

    def eio(path, *a, **kw):  # the Drive read of the zip fails mid-way
        if Path(path) == zip0:
            raise OSError(5, "Input/output error")
        return real_sha(path, *a, **kw)

    B.sha256_file = eio
    try:
        fake = FakeLauncher({"yolo26n_def3_fold0": ["ok"], "yolo26n_def3_fold1": ["ok"]})
        rc, out = _run(_argv(drive, cfgs), fake)
    finally:
        B.sha256_file = real_sha
    assert fake.calls == [("yolo26n_def3_fold1", 1)], (fake.calls, out)
    assert rc == 1 and "NOT retrained" in out, out
    assert not [p for p in bundles.iterdir() if ".invalid-" in p.name]
    assert B.sha256_file(zip0) == sha_before and B.verify_bundle(zip0).ok
    last0 = [r for r in _log(drive) if r["run_name"] == "yolo26n_def3_fold0"][-1]
    assert last0["status"] == "failed" and "Input/output error" in last0["error_tail"], last0
    # an unreadable marker (here: no read permission) is the same case, never "no marker"
    if os.geteuid() != 0:  # root reads anything
        marker = B.sha_path_for(zip0)
        marker.chmod(0)
        try:
            fake = FakeLauncher({"yolo26n_def3_fold0": ["ok"]})
            rc, out = _run(_argv(drive, cfgs[:1]), fake)
        finally:
            marker.chmod(0o644)
        assert rc == 1 and fake.calls == [], out
        assert not [p for p in bundles.iterdir() if ".invalid-" in p.name]
        assert B.verify_bundle(zip0).ok


def test_config_errors_are_not_retried():
    drive, cfgs = _fresh("yolo26n_def3_fold0")
    fake = FakeLauncher({"yolo26n_def3_fold0": ["config", "ok"]})
    rc, out = _run(_argv(drive, cfgs, "--max-attempts", "3"), fake)
    assert rc == 1 and fake.calls == [("yolo26n_def3_fold0", 1)], out
    assert "not retried" in out


def test_bad_configs_fail_fast_before_any_training():
    drive, cfgs = _fresh("yolo26n_def3_fold0", extra_lines="epoch: 3\n")  # typo: epoch
    fake = FakeLauncher({})
    rc, out = _run(_argv(drive, cfgs), fake)
    assert rc == 2 and fake.calls == [] and "epoch" in out, out
    assert _log(drive) == []
    drive, cfgs = _fresh("yolo26n_def3_fold0")
    rc, out = _run(_argv(drive, [cfgs[0], cfgs[0]]), fake)
    assert rc == 2 and "twice" in out, out
    rc, out = _run(_argv(drive, [str(drive / "nope.yaml")]), fake)
    assert rc == 2 and "not found" in out
    rc, out = _run(_argv(drive, cfgs, "--set", "name=x"), fake)
    assert rc == 2 and "identity" in out
    assert fake.calls == []


def test_dry_run_trains_nothing():
    drive, cfgs = _fresh("yolo26n_def3_fold0", "yolo26n_def3_fold1")
    fake = FakeLauncher({})
    rc, out = _run(_argv(drive, cfgs, "--dry-run"), fake)
    assert rc == 0 and fake.calls == [], out
    assert "yolo26n_def3_fold1" in out and "fresh" in out


def test_committed_matrix_configs_load():
    cfgs = sorted(str(p) for p in (ML / "configs" / "matrix").glob("*.yaml")
                  if not p.name.startswith("deim_"))  # DEIM: tests/test_deim_override.py
    items, errors = M.load_items(cfgs, {})
    assert not errors and len(items) == 12, errors
    assert {it.family for it in items} == {"yolo26n_def3", "yolo26s_def3", "yolo11s_def3", "yolo26n_nal1"}
    assert all(it.epochs == 120 and len(it.config_sha) == 64 for it in items)
    # relative paths resolve against ml/ too (the notebook runs from ml/)
    items, errors = M.load_items(["configs/final/yolo26n_def3_devall.yaml"], {})
    assert not errors and items[0].name == "yolo26n_def3_devall"


# ----------------------------------------------------------------- helpers


def test_profiles_only_touch_runtime_knobs():
    assert M.profile_for("auto", "NVIDIA L4") == "L4"
    assert M.profile_for("auto", "Tesla T4") == "T4"
    assert M.profile_for("auto", "NVIDIA A100-SXM4-40GB") == "A100"
    assert M.profile_for("auto", "NVIDIA H100 80GB HBM3") == "generic"
    assert M.profile_for("auto", None) is None
    assert M.profile_for("none", "NVIDIA L4") is None
    assert M.profile_for("T4", None) == "T4"
    assert M.runtime_knobs("L4", 12) == {"workers": 8}
    assert M.runtime_knobs("A100", 12) == {"workers": 12}
    assert M.runtime_knobs("T4", 2) == {"workers": 2, "cache": "disk"}
    assert M.runtime_knobs("L4", 4) == {"workers": 4, "cache": "disk"}
    assert M.runtime_knobs(None, 8) == {}
    for prof in M.PROFILES:
        for cpus in (1, 2, 4, 8, 12, 96):
            assert set(M.runtime_knobs(prof, cpus)) <= {"workers", "cache"}
    # the knobs reach dispatch as runtime flags, never as config overrides
    drive, cfgs = _fresh("yolo26n_def3_fold0")
    args = M.build_parser().parse_args(["--configs", *cfgs, "--drive", str(drive), "--profile", "T4"])
    items, _ = M.load_items(cfgs, {})
    cmd = M.Matrix(args, items, FakeLauncher({}), RUN_ROOT).command(items[0])
    assert "--workers" in cmd and "--set" not in cmd, cmd
    assert not any(k in " ".join(cmd) for k in ("batch", "imgsz", "epochs")), cmd


def test_eta_from_measured_rates():
    book = M.RateBook()
    assert "ETA: unknown" in M.eta_line([M.Item(Path("a"), "yolo26n_def3_fold1", "yolo26n_def3",
                                                "yolo26n", 120, None)], book)
    book.add("yolo26n_def3", "yolo26n", 60.0, 120)  # 0.5 min/epoch
    items = [M.Item(Path("a"), "yolo26n_def3_fold1", "yolo26n_def3", "yolo26n", 120, None),
             M.Item(Path("b"), "yolo26n_nal1_fold0", "yolo26n_nal1", "yolo26n", 120, None),
             M.Item(Path("c"), "yolo11s_def3_fold0", "yolo11s_def3", "yolo11s", 120, None)]
    line = M.eta_line(items, book)
    assert "~2.0 h for 2 of 3" in line and "measured" in line and "from yolo26n" in line, line
    assert "yolo11s_def3_fold0" in line
    seeded = M.RateBook.from_log([{"status": "ok", "run_name": "yolo26s_def3_fold0", "minutes": 90.0,
                                   "epochs_this_attempt": 120},
                                  {"status": "failed", "run_name": "yolo11s_def3_fold0", "minutes": 5.0,
                                   "epochs_this_attempt": 3}])
    assert seeded.rate("yolo26s_def3", "yolo26s")[0] == 0.75
    assert seeded.rate("yolo11s_def3", "yolo11s")[0] is None
    assert M.eta_line([], book) == "ETA: nothing left to run"


def test_attempt_facts_ignore_stale_attempt_logs():
    state = {"attempts": [{"started": "2026-09-28T10:00:00Z", "start_epoch": 0, "end_epoch": 5,
                           "resumed": False}]}
    assert M._attempt_facts(state, "2026-09-28T11:00:00Z", True) == (5, None, True)
    assert M._attempt_facts(state, "2026-09-28T09:00:00Z", True) == (5, 5, False)
    assert M._attempt_facts(None, "2026-09-28T09:00:00Z", False) == (None, None, False)


def test_tee_subprocess_streams_to_log_and_keeps_a_tail():
    d = Path(tempfile.mkdtemp(dir=_TMP_ROOT))
    item = M.Item(Path("x.yaml"), "r", "r", "r", 1, None)
    code = ("import sys\nfor i in range(200): print('line', i)\n"
            "print('progress 1/2', end='\\r'); print('progress 2/2')\n"
            f"print('{dispatch.BUNDLE_MARKER}/tmp/r_bundle.zip'); sys.exit(3)")
    req = M.RunRequest(item, 1, [sys.executable, "-c", code], dict(os.environ), d, d / "logs" / "r.log",
                       d / "mirror", d / "run_dirs")
    out = io.StringIO()
    with redirect_stdout(out):
        res = M.tee_subprocess(req)
    assert res.returncode == 3
    log = (d / "logs" / "r.log").read_text()
    assert "line 0" in log and "line 199" in log and "attempt 1" in log
    assert "line 199" in out.getvalue()
    assert res.tail.splitlines()[-1] == f"{dispatch.BUNDLE_MARKER}/tmp/r_bundle.zip"
    assert "progress 2/2" in res.tail and len(res.tail.splitlines()) <= M.TAIL_LINES
    assert M._bundle_from_tail(res.tail) == Path("/tmp/r_bundle.zip")


def test_tee_survives_a_dead_drive_log():
    # T1: logs/<run>.log on Drive failing (mount dropped) must never stop the child
    d = Path(tempfile.mkdtemp(dir=_TMP_ROOT))
    done = d / "child_finished"
    item = M.Item(Path("x.yaml"), "r", "r", "r", 1, None)
    code = ("import pathlib, sys\nfor i in range(50): print('line', i, flush=True)\n"
            f"pathlib.Path({str(done)!r}).write_text('ok')\nprint('the end'); sys.exit(0)")

    class Dropping:  # accepts the header, then the mount goes away
        def __init__(self):
            self.writes, self.closed = 0, False

        def write(self, data):
            self.writes += 1
            if self.writes > 1:
                raise OSError(107, "Transport endpoint is not connected")

        def flush(self):
            pass

        def close(self):
            self.closed = True

    log = Dropping()
    real_open = getattr(M, "_open_log", None)
    M._open_log = lambda path: log
    try:
        req = M.RunRequest(item, 1, [sys.executable, "-c", code], dict(os.environ), d,
                           d / "logs" / "r.log", d / "mirror", d / "run_dirs")
        out = io.StringIO()
        with redirect_stdout(out):
            res = M.tee_subprocess(req)
    finally:
        M._open_log = real_open
    assert res.returncode == 0 and done.read_text() == "ok", "the child ran to its end"
    assert "the end" in res.tail and "line 49" in out.getvalue()
    assert out.getvalue().count("WARNING") == 1 and log.closed, out.getvalue()
    # a log that cannot even be opened (its folder is a file) is the same case
    blocker = d / "not_a_dir"
    blocker.write_text("x")
    done.unlink()
    req = M.RunRequest(item, 2, [sys.executable, "-c", code], dict(os.environ), d,
                       blocker / "r.log", d / "mirror", d / "run_dirs")
    out = io.StringIO()
    with redirect_stdout(out):
        res = M.tee_subprocess(req)
    assert res.returncode == 0 and done.read_text() == "ok" and "WARNING" in out.getvalue()


def test_publish_refuses_an_invalid_local_bundle():
    d = Path(tempfile.mkdtemp(dir=_TMP_ROOT))
    bad = d / "r_bundle.zip"
    bad.write_bytes(b"not a zip")
    try:
        M.publish_bundle(bad, d / "bundles")
    except RuntimeError as e:
        assert "invalid" in str(e)
    else:
        raise AssertionError("an invalid bundle was published")
    assert not (d / "bundles" / "r_bundle.zip").exists()


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
    sys.exit(1 if failed else 0)
