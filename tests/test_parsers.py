"""
Unit tests for the load-bearing Label Studio parsers used by the dashboard.

These map an LS task back to a capture_id and pull the overall_quality choice
out of an annotation — both are relied on by the live "labeled" view, so a
regression here silently breaks the dashboard.

Run (no pytest needed):  python tests/test_parsers.py
"""
import os
import sys
import tempfile
from pathlib import Path

# Keep imports side-effect-free of the real home dir.
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="zander-test-"))
os.environ.setdefault("LABELSTUDIO_API_KEY", "")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))

import main as dash  # noqa: E402


def test_capture_id_from_meta():
    task = {"meta": {"capture_id": "abc123"}, "data": {"image": "/data/local-files/?d=unlabeled/zzz.jpg"}}
    assert dash._task_capture_id(task) == "abc123"


def test_capture_id_from_image_url():
    task = {"meta": {}, "data": {"image": "/data/local-files/?d=unlabeled/cf3e0b64.jpg"}}
    assert dash._task_capture_id(task) == "cf3e0b64"


def test_capture_id_missing():
    assert dash._task_capture_id({"meta": {}, "data": {}}) is None


def test_overall_quality_io():
    result = [
        {"from_name": "label", "value": {"rectanglelabels": ["Brücke"]}},
        {"from_name": "overall_quality", "value": {"choices": ["iO"]}},
    ]
    assert dash._extract_overall_quality(result) == "iO"


def test_overall_quality_nio():
    result = [{"from_name": "overall_quality", "value": {"choices": ["NiO"]}}]
    assert dash._extract_overall_quality(result) == "NiO"


def test_overall_quality_absent():
    assert dash._extract_overall_quality([{"from_name": "label", "value": {}}]) is None
    assert dash._extract_overall_quality([]) is None


def test_completed_by_email_object_and_id():
    # Object form (webhook payload shape)
    assert dash._completed_by_email({"completed_by": {"email": "a@b.c"}}) == "a@b.c"
    # Int form (REST list shape) → resolved via the LS client; unreachable here,
    # so it returns None without raising.
    assert dash._completed_by_email({"completed_by": 999999}) is None


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
