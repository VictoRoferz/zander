"""
scripts/launch.py: the optional inspector must never take the hub down
(docs/CONTRACTS.md section 7), plus the helpers that pick its interpreter and
decide whether the receiver gets INSPECTOR_URL.

Run (from the repo root): ../venv/bin/python -B tests/test_launch_helpers.py
Stdlib only; any Python 3.9+ works. The end-to-end case runs launch.main()
against throwaway stand-in services (tiny python scripts), so no Label
Studio, receiver or model is needed.
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location("launch", ROOT / "scripts" / "launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class FakeProc:
    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code


def _capture(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn()
    return result, buf.getvalue()


# ---- first_critical_exit -------------------------------------------------------
def test_nothing_exited():
    procs = [("receiver", FakeProc(), True), ("inspector", FakeProc(), False)]
    result, out = _capture(lambda: launch.first_critical_exit(procs, set()))
    assert result is None and out == ""


def test_non_critical_exit_is_logged_once_and_ignored():
    procs = [("receiver", FakeProc(), True), ("inspector", FakeProc(3), False)]
    reported = set()
    result, out = _capture(lambda: launch.first_critical_exit(procs, reported))
    assert result is None
    assert "inspector exited with code 3" in out and "optional" in out
    assert reported == {"inspector"}
    result, out = _capture(lambda: launch.first_critical_exit(procs, reported))
    assert result is None and out == "", "logged twice"


def test_critical_exit_stops_even_after_a_non_critical_one():
    procs = [
        ("label-studio", FakeProc(), True),
        ("inspector", FakeProc(1), False),
        ("receiver", FakeProc(0), True),
    ]
    result, _ = _capture(lambda: launch.first_critical_exit(procs, set()))
    assert result == ("receiver", 0)


# ---- interpreter + INSPECTOR_URL helpers ---------------------------------------------
def test_explicit_inspector_python_that_does_not_exist_is_skipped():
    python, why = launch.find_inspector_python({"INSPECTOR_PYTHON": "/nonexistent/python3"})
    assert python is None and "INSPECTOR_PYTHON" in why and "not runnable" in why, why


def test_explicit_inspector_python_is_checked_for_the_inference_stack():
    has_deps = all(
        importlib.util.find_spec(m) is not None for m in ("onnxruntime", "PIL", "fastapi")
    )
    python, why = launch.find_inspector_python({"INSPECTOR_PYTHON": sys.executable})
    if has_deps:
        assert python == sys.executable, why
    else:
        assert python is None and "lacks the inspector deps" in why, why


def test_missing_ml_venv_is_skipped():
    saved = launch.ml_venv_python
    launch.ml_venv_python = lambda: Path("/nonexistent/ml/.venv/bin/python")
    try:
        python, why = launch.find_inspector_python({})
    finally:
        launch.ml_venv_python = saved
    assert python is None and "no ml/.venv interpreter" in why, why


def test_inspector_url_configured():
    with tempfile.TemporaryDirectory() as d:
        env_file = Path(d) / ".env"
        assert not launch.inspector_url_configured({}, env_file)  # no file
        env_file.write_text("LABELSTUDIO_API_KEY=x\n#INSPECTOR_URL=http://a\n", encoding="utf-8")
        assert not launch.inspector_url_configured({}, env_file), "a comment is not a setting"
        env_file.write_text("INSPECTOR_URL=\n", encoding="utf-8")
        assert launch.inspector_url_configured({}, env_file), "explicit empty = off, keep it"
        env_file.write_text("export inspector_url = http://b\n", encoding="utf-8")
        assert launch.inspector_url_configured({}, env_file)
        empty = Path(d) / "none.env"
        assert launch.inspector_url_configured({"INSPECTOR_URL": ""}, empty)
        assert launch.inspector_url_configured({"inspector_url": "http://c"}, empty)
        assert not launch.inspector_url_configured({"PATH": "/bin"}, empty)


# ---- End to end: a crashing inspector leaves the hub running ---------------------------
_ENV_DUMP = (
    "import json, os, sys, time\n"
    "open(os.environ['FAKE_OUT'] + '/{name}.json', 'w').write(json.dumps(dict(os.environ)))\n"
)
FAKE_SERVICES = {
    "receiver": _ENV_DUMP + "time.sleep(float(os.environ['FAKE_RECEIVER_S']))\n",
    "dashboard": _ENV_DUMP + "time.sleep(30)\n",
    "camerapi": _ENV_DUMP + "time.sleep(30)\n",
    "inspector": _ENV_DUMP + "print('inspector crashed on purpose'); sys.exit(3)\n",
}


def _run_launcher(tmp: Path, argv: list, inspector_python: str, receiver_s: float = 1.6):
    """launch.main() against stand-in services; returns (exit code, output, env dumps, seconds)."""
    out_dir = tmp / "out"
    out_dir.mkdir(exist_ok=True)
    dirs = {}
    for name, body in FAKE_SERVICES.items():
        d = tmp / name
        d.mkdir(exist_ok=True)
        (d / "main.py").write_text(body.replace("{name}", name), encoding="utf-8")
        dirs[name] = d

    saved = {k: getattr(launch, k) for k in (
        "RECEIVER_DIR", "DASHBOARD_DIR", "CAMERAPI_DIR", "INSPECTOR_DIR",
        "label_studio_args", "INSPECTOR_IMPORT_CHECK")}
    saved_env = {k: os.environ.get(k) for k in (
        "DATA_ROOT", "FAKE_OUT", "FAKE_RECEIVER_S", "INSPECTOR_PYTHON", "INSPECTOR_URL")}
    saved_argv = sys.argv
    try:
        launch.RECEIVER_DIR = dirs["receiver"]
        launch.DASHBOARD_DIR = dirs["dashboard"]
        launch.CAMERAPI_DIR = dirs["camerapi"]
        launch.INSPECTOR_DIR = dirs["inspector"]
        launch.label_studio_args = lambda: [sys.executable, "-c", "import time; time.sleep(30)"]
        launch.INSPECTOR_IMPORT_CHECK = "import sys"  # the stand-in needs no ML stack
        launch.PROCS.clear()
        os.environ["DATA_ROOT"] = str(tmp / "data")
        os.environ["FAKE_OUT"] = str(out_dir)
        os.environ["FAKE_RECEIVER_S"] = str(receiver_s)
        os.environ["INSPECTOR_PYTHON"] = inspector_python
        os.environ.pop("INSPECTOR_URL", None)
        sys.argv = ["launch.py", *argv]
        t0 = time.monotonic()
        code, out = _capture(launch.main)
        elapsed = time.monotonic() - t0
    finally:
        sys.argv = saved_argv
        for k, v in saved.items():
            setattr(launch, k, v)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for _name, proc, _critical in launch.PROCS:  # never leak a child
            if proc.poll() is None:
                proc.kill()
        launch.PROCS.clear()
    envs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in out_dir.glob("*.json")}
    return code, out, envs, elapsed


def test_crashing_inspector_leaves_the_hub_running():
    tmp = Path(tempfile.mkdtemp(prefix="zander-launch-test-"))
    try:
        code, out, envs, elapsed = _run_launcher(
            tmp, ["--with-camera", "--no-listener"], inspector_python=sys.executable)
        assert (tmp / "data" / "predictions").is_dir()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert "inspector exited with code 3; it is optional" in out, out
    assert "receiver exited with code 0; stopping the rest." in out, out
    assert code == 1 and elapsed >= 1.4, (code, elapsed)  # the receiver's own exit ended it
    assert envs["inspector"]["DATA_ROOT"] == str(tmp / "data")
    assert envs["receiver"]["INSPECTOR_URL"] == "http://localhost:8004"
    assert "INSPECTOR_URL" not in envs["dashboard"] and "INSPECTOR_URL" not in envs["camerapi"]
    assert all(ord(ch) < 128 for line in out.splitlines() if line.startswith("[launch]")
               for ch in line), "launcher log lines must stay ASCII"


def test_skipped_inspector_gets_no_url_and_hub_runs():
    tmp = Path(tempfile.mkdtemp(prefix="zander-launch-test-"))
    try:
        code, out, envs, _ = _run_launcher(
            tmp, [], inspector_python=str(tmp / "no-such-python"), receiver_s=0.8)
        assert (tmp / "data" / "predictions").is_dir()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert "inspector skipped:" in out, out
    assert "inspector" not in envs and "INSPECTOR_URL" not in envs["receiver"]
    assert "receiver exited with code 0; stopping the rest." in out and code == 1


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
