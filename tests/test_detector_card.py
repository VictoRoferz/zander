"""
detector.json v1 (ml/infer/detector_card.py) — the card shared by `cli.py
export` (writer) and the inspector (reader). See docs/CONTRACTS.md.

Run: ml/.venv/bin/python -B tests/test_detector_card.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from infer.detector_card import (  # noqa: E402
    KNOWN_ADAPTERS,
    DetectorCardError,
    card_path_for,
    load_detector_card,
    sha256_file,
    validate_card,
    write_detector_card,
)
from infer.tiling import TILER_VERSION  # noqa: E402

NAMES = ["Nicht ausreichend Lot", "Brücke", "Fahne"]


def _model(tmp: Path, content: bytes = b"fake onnx bytes") -> Path:
    p = tmp / "detector.onnx"
    p.write_bytes(content)
    return p


def _doc(model: Path, **over) -> dict:
    doc = {
        "contract_version": 1,
        "kind": "zander-detector",
        "model_file": model.name,
        "model_sha256": sha256_file(model),
        "adapter": "ultralytics",
        "schema": "def3",
        "class_names": list(NAMES),
        "threshold": 0.34,
        "conf_floor": 0.34,
        "tiler_version": TILER_VERSION,
        "tile_size": 640,
        "stride": 512,
        "verifier": None,
    }
    doc.update(over)
    return doc


def _raises(fn, *needles):
    try:
        fn()
    except DetectorCardError as e:
        for n in needles:
            assert n in str(e), f"{n!r} not in error: {e}"
        return
    raise AssertionError("expected DetectorCardError")


def test_round_trip_and_card_path():
    with tempfile.TemporaryDirectory() as d:
        model = _model(Path(d))
        written = write_detector_card(card_path_for(model), _doc(model))
        assert written == Path(d) / "detector.json"
        card = load_detector_card(model)
        assert card.class_names == tuple(NAMES)
        assert card.threshold == 0.34 and card.conf_floor == 0.34
        assert card.adapter == "ultralytics" and card.schema == "def3"
        assert card.verifier is None and card.raw["kind"] == "zander-detector"
        assert not [p for p in Path(d).iterdir() if p.name.endswith(".tmp")], "temp file left behind"


def test_sha_mismatch_is_rejected_on_load():
    with tempfile.TemporaryDirectory() as d:
        model = _model(Path(d))
        write_detector_card(card_path_for(model), _doc(model))
        model.write_bytes(b"a different model")
        _raises(lambda: load_detector_card(model), "model_sha256 mismatch")
        load_detector_card(model, verify_sha=False)  # explicit opt-out still loads


def test_writer_refuses_invalid_cards():
    with tempfile.TemporaryDirectory() as d:
        model = _model(Path(d))
        path = card_path_for(model)
        _raises(lambda: write_detector_card(path, _doc(model, adapter="yolo")), "adapter")
        _raises(lambda: write_detector_card(path, _doc(model, threshold=1.0)), "threshold")
        _raises(lambda: write_detector_card(path, _doc(model, conf_floor=0.5)), "conf_floor must be <= threshold")
        _raises(lambda: write_detector_card(path, _doc(model, class_names=["A", "A"])), "class_names")
        _raises(lambda: write_detector_card(path, _doc(model, tiler_version="v1")), "tiler_version")
        _raises(lambda: write_detector_card(path, _doc(model, stride=640)), "stride")
        _raises(lambda: write_detector_card(path, _doc(model, contract_version=2)), "contract_version")
        assert not path.exists(), "an invalid card must never be written"


def test_verifier_requires_matching_candidate_floor():
    with tempfile.TemporaryDirectory() as d:
        model = _model(Path(d))
        ok = _doc(model, conf_floor=0.001, verifier={"candidate_floor": 0.001, "top_k": 50})
        validate_card(ok, model_path=model)
        bad = _doc(model, conf_floor=0.34, verifier={"candidate_floor": 0.001})
        _raises(lambda: validate_card(bad, model_path=model), "candidate_floor")


def test_card_must_name_its_model_file():
    with tempfile.TemporaryDirectory() as d:
        model = _model(Path(d))
        doc = _doc(model, model_file="other.onnx")
        _raises(lambda: validate_card(doc, model_path=model), "card names")
        _raises(lambda: validate_card(_doc(model, model_file="sub/detector.onnx")), "bare file name")


def test_missing_or_unreadable_card():
    with tempfile.TemporaryDirectory() as d:
        model = _model(Path(d))
        _raises(lambda: load_detector_card(model), "card not found")
        card_path_for(model).write_text("{not json", encoding="utf-8")
        _raises(lambda: load_detector_card(model), "card unreadable")


def test_known_adapters_match_onnx_adapters():
    try:
        from infer.onnx_adapters import ADAPTERS
    except ImportError:
        print("SKIP adapter parity (onnxruntime missing)")
        return
    assert set(KNOWN_ADAPTERS) == set(ADAPTERS), (KNOWN_ADAPTERS, sorted(ADAPTERS))


def test_card_is_plain_json():
    with tempfile.TemporaryDirectory() as d:
        model = _model(Path(d))
        write_detector_card(card_path_for(model), _doc(model))
        text = card_path_for(model).read_text(encoding="utf-8")
        assert "Brücke" in text, "class names must be stored as UTF-8, not escaped"
        assert json.loads(text)["model_file"] == "detector.onnx"


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
