"""
The Colab notebook (ml/notebooks/colab_train.ipynb) is valid and points at
things that exist: JSON parses, the parameters are there, every referenced
config exists, every code cell compiles (IPython `!`/`%` lines stubbed), the
matrix is never run with check=True, and the v2 guard is present.

Run: ml/.venv/bin/python -B tests/test_notebook.py
"""
import ast
import io
import json
import os
import re
import shutil
import sys
import tempfile
import types
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "ml"
NB = ML / "notebooks" / "colab_train.ipynb"
RUNBOOK = ML / "notebooks" / "README.md"

EXPECTED_MATRIX = [  # priority order (n-scale def3 first, then its nal1 comparison, then s-scale)
    "configs/matrix/yolo26n_def3_fold0.yaml",
    "configs/matrix/yolo26n_def3_fold1.yaml",
    "configs/matrix/yolo26n_def3_fold2.yaml",
    "configs/matrix/yolo26n_nal1_fold0.yaml",
    "configs/matrix/yolo26n_nal1_fold1.yaml",
    "configs/matrix/yolo26n_nal1_fold2.yaml",
    "configs/matrix/yolo26s_def3_fold0.yaml",
    "configs/matrix/yolo26s_def3_fold1.yaml",
    "configs/matrix/yolo26s_def3_fold2.yaml",
    "configs/matrix/yolo11s_def3_fold0.yaml",
    "configs/matrix/yolo11s_def3_fold1.yaml",
    "configs/matrix/yolo11s_def3_fold2.yaml",
]


def _nb() -> dict:
    return json.loads(NB.read_text(encoding="utf-8"))


def _src(cell) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _code_cells():
    return [c for c in _nb()["cells"] if c["cell_type"] == "code"]


def _params() -> dict:
    cells = [c for c in _code_cells() if "parameters" in c.get("metadata", {}).get("tags", [])]
    assert len(cells) == 1, "exactly one code cell tagged 'parameters'"
    ns: dict = {}
    exec(compile(_src(cells[0]), "<parameters>", "exec"), ns)  # must be plain Python
    return ns


def _stub_ipython(src: str) -> str:
    """Replace IPython-only lines (`!cmd`, `%magic`) by `pass`, keeping indentation."""
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            out.append(line[: len(line) - len(stripped)] + "pass")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def test_notebook_json_is_valid():
    nb = _nb()
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 4
    ids = [c.get("id") for c in nb["cells"]]
    assert all(ids) and len(set(ids)) == len(ids), "nbformat 4.5 needs unique cell ids"
    for c in nb["cells"]:
        assert c["cell_type"] in ("code", "markdown")
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None, "commit the notebook without outputs"
    assert nb["metadata"].get("accelerator") == "GPU"


def test_parameters():
    p = _params()
    assert p["BRANCH"] == "feature/ds-v2-preprocessing"
    assert p["EXPECT_GPU"] == "L4"
    assert p["MATRIX"] == EXPECTED_MATRIX, p["MATRIX"]
    assert p["FINAL"] is None
    assert p["EXTRA_SET"] == []
    assert p["DRIVE_DIR"].endswith("/ZanderML")
    assert p["PROFILE"] in ("auto", "L4", "A100", "T4", "none")
    assert isinstance(p["MAX_ATTEMPTS"], int) and p["MAX_ATTEMPTS"] >= 1


def test_every_referenced_config_exists():
    p = _params()
    for cfg in p["MATRIX"] + (p["FINAL"] or []):
        assert (ML / cfg).is_file(), cfg
    text = NB.read_text(encoding="utf-8")
    mentioned = set(re.findall(r"configs/[A-Za-z0-9_./-]+\.yaml", text))
    assert "configs/final/yolo26n_def3_devall.yaml" in mentioned  # the FINAL example
    for cfg in mentioned:
        assert (ML / cfg).is_file(), f"notebook mentions a missing config: {cfg}"


def test_code_cells_compile():
    for c in _code_cells():
        src = _stub_ipython(_src(c))
        try:
            ast.parse(src)
        except SyntaxError as e:
            raise AssertionError(f"cell {c.get('id')}: {e}\n{src}")


def test_no_check_true_and_matrix_driver_used():
    code = "\n".join(_src(c) for c in _code_cells())
    assert "check=True" not in code, "a failing run must not kill the notebook loop"
    assert "train/matrix.py" in code and "--drive" in code and "--max-attempts" in code
    assert "cli.py train" not in code, "training goes through train/matrix.py"
    assert "requirements-colab.txt" in code
    assert "requirements.txt" not in code, "Colab must not install the pinned numpy/Pillow of requirements.txt"


def test_v2_guard_and_fingerprint_assert():
    code = "\n".join(_src(c) for c in _code_cells())
    assert re.search(r"assert\s+os\.path\.exists\([^)]*ml/datasets/split_v2\.json", code), "v2 guard missing"
    assert "dataset_fingerprint" in code and re.search(r"assert\s+rebuilt\s*==\s*fp", code)
    assert "--depth 1" in code, "shallow clone"
    assert "rev-parse" in code, "print the git sha"


def test_session_ends_with_a_drive_flush():
    # T5: Drive uploads in the background; the last code cell waits for it
    code = _code_cells()
    assert code[-1]["id"] == "flush", [c["id"] for c in code]
    assert "drive.flush_and_unmount()" in _src(code[-1])
    md = "\n".join(_src(c) for c in _nb()["cells"] if c["cell_type"] == "markdown")
    assert "deleted once a bundle is safe" not in md and "deletes the run's mirror" not in md
    assert "later" in md and "flush_and_unmount" in RUNBOOK.read_text(encoding="utf-8")


def _cell(cell_id: str) -> str:
    return _src({c["id"]: c for c in _nb()["cells"]}[cell_id])


def test_colab_pins_have_wheels_for_current_pythons():
    # T3: onnx==1.17.0 has wheels for cp38-cp312 only; 1.19.1 has cp310-cp313 manylinux wheels
    from packaging.requirements import Requirement

    reqs = {}
    for line in (ML / "requirements-colab.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            r = Requirement(line)
            reqs[r.name] = r.specifier
    assert reqs["onnx"].contains("1.19.1"), reqs["onnx"]
    assert not reqs["onnx"].contains("1.20.0"), "keep the onnx line the verified ultralytics/ORT combo"
    assert str(reqs["ultralytics"]) == "==8.4.115"


def test_a_failed_install_stops_run_all():
    # T3: `%pip` never fails a cell — the cell after it must import the stack or raise
    order = [c["id"] for c in _code_cells()]
    assert order.index("install") + 1 == order.index("deps") < order.index("data"), order
    src = _cell("deps")
    assert not any(line.lstrip().startswith(("!", "%")) for line in src.splitlines())
    saved = {m: sys.modules.get(m) for m in ("ultralytics", "onnxslim")}
    sys.modules["ultralytics"] = types.SimpleNamespace(__version__="8.4.115")  # no torch import in this test
    try:
        with redirect_stdout(io.StringIO()):
            exec(compile(src, "<deps>", "exec"), {})  # the ml venv has the stack: passes
        sys.modules["onnxslim"] = None  # -> ImportError, like a package pip could not install
        try:
            with redirect_stdout(io.StringIO()):
                exec(compile(src, "<deps>", "exec"), {})
        except RuntimeError as e:
            assert "onnxslim" in str(e) and "requirements-colab.txt" in str(e), e
        else:
            raise AssertionError("a missing package did not stop the notebook")
    finally:
        for m, mod in saved.items():
            if mod is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = mod
    assert "python_version" in src


def _run_data_cell(cli_code: str, zip_fp: str = "a" * 64):
    """Execute the notebook's data cell against a fake repo/Drive whose
    `cli.py tile` runs `cli_code` (ML_DATA_ROOT is set for it)."""
    d = Path(tempfile.mkdtemp(prefix="zander_test_nb_"))
    try:
        repo, drive, ml_data = d / "repo", d / "drive", d / "ml_data"
        (repo / "ml" / "configs").mkdir(parents=True)
        (repo / "ml" / "configs" / "fake_def3_fold0.yaml").write_text("dataset_id: ds_v2_def3\n")
        (repo / "ml" / "cli.py").write_text(cli_code)
        drive.mkdir()
        with zipfile.ZipFile(drive / "ds_v2_def3_tiles.zip", "w") as zf:
            zf.writestr("ds_v2_def3/provenance.json",
                        json.dumps({"dataset_fingerprint": zip_fp, "schema": "def3"}))
        ns = {"os": os, "REPO": str(repo), "BRANCH": "test", "ML_DATA": str(ml_data),
              "DRIVE_DIR": str(drive), "MATRIX": ["configs/fake_def3_fold0.yaml"], "FINAL": None,
              "DEIM_MATRIX": []}
        saved = os.environ.get("ML_DATA_ROOT")
        try:
            with redirect_stdout(io.StringIO()):
                exec(compile(_stub_ipython(_cell("data")), "<data>", "exec"), ns)
        finally:
            if saved is None:
                os.environ.pop("ML_DATA_ROOT", None)
            else:
                os.environ["ML_DATA_ROOT"] = saved
    finally:
        shutil.rmtree(d, ignore_errors=True)


_WRITE_PROV = ("import json, os, pathlib, sys\n"
               "p = pathlib.Path(os.environ['ML_DATA_ROOT']) / 'tiles' / 'ds_v2_def3' / 'provenance.json'\n"
               "p.write_text(json.dumps({'dataset_fingerprint': %r}))\n")


def test_the_fingerprint_check_cannot_pass_on_a_failed_rebuild():
    # T4: the rebuild used to be `!python cli.py tile` (never raises) followed by
    # comparing the zip's own provenance.json with itself
    assert "cli.py tile" not in "".join(l for l in _cell("data").splitlines(True) if l.lstrip().startswith("!"))
    _run_data_cell(_WRITE_PROV % ("a" * 64))  # same data + code: passes
    for code, needle in (("import sys\nsys.exit(1)\n", "rebuilding the YOLO view failed"),
                         ("print('did nothing')\n", "rebuilding the YOLO view failed")):
        try:
            _run_data_cell(code)
        except RuntimeError as e:
            assert needle in str(e), e
        else:
            raise AssertionError(f"a failed rebuild passed the fingerprint check: {code!r}")
    try:
        _run_data_cell(_WRITE_PROV % ("b" * 64))
    except AssertionError as e:
        assert "fingerprint mismatch" in str(e), e
    else:
        raise AssertionError("a different fingerprint passed")


def test_summary_survives_a_torn_log_line():
    # the summary cell runs right before the Drive flush: if it raised, Run all
    # would stop before flushing (a VM that died mid-append leaves a torn line)
    d = Path(tempfile.mkdtemp(prefix="zander_test_nb_"))
    try:
        (d / "matrix_log.jsonl").write_text(
            json.dumps({"run_name": "yolo26n_def3_fold0", "status": "ok", "minutes": 60.0,
                        "epochs_completed": 120, "ended": "2026-10-09T10:00:00Z"}) + "\n"
            + '{"run_name": "yolo26n_def3_fold1", "sta')
        ns = {"DRIVE_DIR": str(d), "REPO": str(ROOT), "MATRIX": [], "FINAL": None, "DEIM_MATRIX": []}
        out = io.StringIO()
        with redirect_stdout(out):
            exec(compile(_stub_ipython(_cell("summary")), "<summary>", "exec"), ns)
        assert "yolo26n_def3_fold0" in out.getvalue() and "1.0 GPU-hours" in out.getvalue(), out.getvalue()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_markdown_states_honest_timings():
    md = "\n".join(_src(c) for c in _nb()["cells"] if c["cell_type"] == "markdown")
    for needle in ("50–90 min", "35 %", "11–17 h", "55–85 compute units", "2 sessions",
                   "Reconnect", "Run all", "skipped", "resumes"):
        assert needle in md, needle
    assert "10 GPU-min" not in md and "~2-2.5 h" not in md, "the old, wrong timings are gone"


DEIM_CONFIGS = [
    "configs/matrix/deim_s_def3_fold0.yaml",
    "configs/matrix/deim_s_def3_fold1.yaml",
    "configs/matrix/deim_s_def3_fold2.yaml",
]


def test_optional_deim_step():
    p = _params()
    assert p["DEIM_MATRIX"] == [], "DEIM is optional: empty by default"
    assert p["MATRIX"] == EXPECTED_MATRIX and not any("deim" in c for c in p["MATRIX"])
    text = NB.read_text(encoding="utf-8")
    for cfg in DEIM_CONFIGS + ["configs/final/deim_s_def3_devall.yaml"]:
        assert cfg in text and (ML / cfg).is_file(), cfg  # the example in the parameters cell
    cells = {c["id"]: c for c in _nb()["cells"]}
    order = [c["id"] for c in _nb()["cells"]]
    assert order.index("matrix") < order.index("deim_md") < order.index("deim") < order.index("final"), order
    deim = _src(cells["deim"])
    assert "if DEIM_NEEDED:" in deim and "if DEIM_MATRIX:" in deim and "_is_deim" in deim
    assert "train/deim_run.py --install-deps" in deim and "train/deim_run.py --fetch-weights" in deim
    assert "train/matrix.py --configs {DEIM_ARGS}" in deim and "--drive" in deim and "--max-attempts" in deim
    for line in deim.splitlines():  # nothing is installed or downloaded unless DEIM is used
        if "--install-deps" in line or "--fetch-weights" in line:
            assert line.startswith("    !"), f"unguarded: {line}"
    assert "requirements.txt" not in deim and "%pip" not in deim and "check=True" not in deim
    assert "CONFIGS = MATRIX + (FINAL or []) + DEIM_MATRIX" in _src(cells["data"])
    assert "DEIM_MATRIX" in _src(cells["summary"])
    md = _src(cells["deim_md"])
    for needle in ("Optional", "Apache-2.0", "DEIMv2", "estimate", "1.5–3.5 h", "25–50 compute units",
                   "LICENSE_DEIM"):
        assert needle in md, needle
    runbook = RUNBOOK.read_text(encoding="utf-8")
    for needle in ("Optional: DEIM", "DEIM_MATRIX", "--install-deps", "--fetch-weights", "1.5–3.5 h",
                   "MyDrive/ZanderML/pretrained/deim_hgnetv2_s_coco.pth", "LICENSE_DEIM",
                   "configs/deim_smoke_def3.yaml", "--dry-run",
                   # the troubleshooting rows quote the runner's real messages
                   "is not the init checkpoint of deim_hgnetv2_s_coco", "fills only N/M tensors",
                   "transformers, accelerate, huggingface_hub", ".stale-"):
        assert needle in runbook, needle
    assert "!= recorded" not in runbook and "pinned on first use" not in text, \
        "the init checkpoint's sha256 is pinned in train/deim_run.py, not recorded on first use"


def test_runbook_exists_and_covers_the_steps():
    text = RUNBOOK.read_text(encoding="utf-8")
    for needle in ("cli.py tile --schema def3 --zip", "cli.py tile --schema nal1 --zip",
                   "MyDrive/ZanderML", "colab.research.google.com/github/VictoRoferz/zander/blob/",
                   "High-RAM", "L4", "supervisor", "matrix_log.jsonl", "Run all"):
        assert needle in text, needle


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
