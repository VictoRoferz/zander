"""
Inspector service (inspector/): detector.json loading, the legacy env
fallback, the refusal rules, the /inspect response shape, the models folder
(several models, the active one, the saved choice) and the HTTP contract
(docs/CONTRACTS.md sections 1 and 3).

Run: ml/.venv/bin/python -B tests/test_inspector_service.py
Skips (exit 0) when onnxruntime / Pillow / fastapi are missing.

Most cases replace TiledDetector with a fake (no real model needed); one case
builds a tiny real ONNX graph (needs the `onnx` package, skipped otherwise) so
the real TiledDetector + adapter wiring is exercised too. HTTP cases drive
the ASGI app in-process: fastapi's TestClient needs httpx, which ml/.venv
does not ship.
"""
import asyncio
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    import fastapi  # noqa: F401
    import numpy as np
    import onnxruntime  # noqa: F401
    from PIL import Image
except ImportError as e:
    print(f"SKIP: {e} (run with ml/.venv/bin/python)")
    sys.exit(0)

# ---- Environment BEFORE importing the service (settings are read at import) --
TMP = Path(tempfile.mkdtemp(prefix="zander-inspector-test-"))
DATA = TMP / "data"
MODEL = TMP / "models" / "detector.onnx"
os.environ.update({
    "DATA_ROOT": str(DATA),
    "MODEL_PATH": str(MODEL),
    "REQUIRE_CARD": "false",
    "LOG_LEVEL": "CRITICAL",
})
for key in ("ADAPTER", "CONFIDENCE_THRESHOLD", "CLASS_NAMES", "ORT_THREADS"):
    os.environ.pop(key, None)
os.chdir(TMP)  # the service reads ./.env; make sure there is none
sys.path.insert(0, str(ROOT / "inspector"))

import main  # noqa: E402
import services.detector_service as ds  # noqa: E402
from config.settings import DEFAULT_CLASS_NAMES, settings  # noqa: E402
from infer.detector_card import card_path_for, sha256_file, write_detector_card  # noqa: E402
from infer.pipeline import Detections  # noqa: E402
from infer.pipeline import TiledDetector as RealTiledDetector  # noqa: E402
from infer.tiling import TILER_VERSION  # noqa: E402

detector_service = ds.detector_service
NAMES = ["Nicht ausreichend Lot", "Brücke", "Fahne"]

INSPECT_KEYS = {
    # kept from inspector v0.1
    "status", "source", "verdict", "classes", "detections", "count", "latency_ms",
    "model_version", "tiler_version", "confidence_threshold",
    # added by CONTRACTS section 3
    "photo_score", "image_width", "image_height", "model_sha256", "schema",
    "adapter", "class_names", "threshold_source",
}
STATUS_KEYS = {
    "service", "version", "model_loaded", "active_model", "model_path", "meta_path", "model_version",
    "model_sha256", "adapter", "schema", "class_names", "confidence_threshold",
    "threshold_source", "tiler_version", "load_error", "meta",
}
MODELS_KEYS = {
    "active", "model_loaded", "load_error", "selected_by", "selected_at", "models_dir", "models",
}
MODEL_ENTRY_KEYS = {
    "name", "loaded", "active", "error", "model_version", "model_sha256", "adapter", "schema",
    "class_names", "threshold", "threshold_source", "run_name", "created",
}


# ---- Log capture (the WARNINGs are part of the contract) ---------------------
class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


LOGS = _ListHandler()
_det_logger = logging.getLogger("inspector.detector")
_det_logger.setLevel(logging.DEBUG)
_det_logger.addHandler(LOGS)
_det_logger.propagate = False


# ---- Fake TiledDetector -------------------------------------------------------
class FakeAdapter:
    def __init__(self, num_classes):
        self.num_classes = num_classes


class FakeTiledDetector:
    num_classes = 3  # what the fake model reports (None = not knowable)
    raise_on_init = None
    detect_sleep_s = 0.05
    instances: list = []
    active = 0
    max_active = 0
    counter_lock = threading.Lock()
    # labels 0/1/2 = NAL 0.5, Brücke 0.9, Fahne 0.1 (below every test threshold)
    boxes = [(10.0, 10.0, 40.0, 40.0), (100.0, 100.0, 150.0, 150.0), (5.0, 5.0, 8.0, 8.0)]
    scores = [0.5, 0.9, 0.1]
    labels = [0, 1, 2]

    def __init__(self, model_path, adapter_name="ultralytics", conf_floor=0.001, threads=0):
        if FakeTiledDetector.raise_on_init is not None:
            raise FakeTiledDetector.raise_on_init
        self.model_path = model_path
        self.adapter_name = adapter_name
        self.conf_floor = conf_floor
        self.threads = threads
        self.adapter = FakeAdapter(FakeTiledDetector.num_classes)
        self.tiler_version = TILER_VERSION
        FakeTiledDetector.instances.append(self)

    def detect(self, photo_rgb):
        cls = FakeTiledDetector
        with cls.counter_lock:
            cls.active += 1
            cls.max_active = max(cls.max_active, cls.active)
            concurrent = cls.active
        try:
            assert concurrent == 1, f"{concurrent} concurrent detect() calls"
            time.sleep(cls.detect_sleep_s)
            return Detections(list(cls.boxes), list(cls.scores), list(cls.labels), 50.0)
        finally:
            with cls.counter_lock:
                cls.active -= 1


ds.TiledDetector = FakeTiledDetector


# ---- Helpers --------------------------------------------------------------------
def _reset(**settings_over):
    ds.TiledDetector = FakeTiledDetector
    FakeTiledDetector.num_classes = 3
    FakeTiledDetector.raise_on_init = None
    FakeTiledDetector.detect_sleep_s = 0.05
    FakeTiledDetector.instances.clear()
    FakeTiledDetector.max_active = 0
    settings.model_path = MODEL
    settings.models_dir = None
    settings.require_card = False
    settings.adapter = None
    settings.confidence_threshold = None
    settings.class_names = None
    for key, value in settings_over.items():
        setattr(settings, key, value)
    shutil.rmtree(MODEL.parent, ignore_errors=True)
    MODEL.parent.mkdir(parents=True)
    LOGS.messages.clear()


def _model(content: bytes = b"fake onnx bytes") -> Path:
    MODEL.write_bytes(content)
    return MODEL


def _card(model: Path = MODEL, **over) -> Path:
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
        "decision_rule": "NiO iff any detection score >= threshold (inclusive)",
        "tiler_version": TILER_VERSION,
        "tile_size": 640,
        "stride": 512,
        "source": {
            "bundle_run": "yolo26n_def3_devall",
            "split_id": "split_v2",
            "dataset_fingerprint": "fp123",
        },
        "verifier": None,
        "exported_at": "2026-09-28T10:00:00Z",
    }
    doc.update(over)
    return write_detector_card(card_path_for(model), doc)


def _named_model(name: str, card: bool = True, folder: Path = None, **card_over) -> Path:
    """Another model in the models folder (distinct bytes = distinct sha)."""
    path = (folder or MODEL.parent) / f"{name}.onnx"
    path.write_bytes(f"fake onnx {name}".encode())
    if card:
        _card(path, **card_over)
    return path


def _jpeg(width: int = 700, height: int = 500) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (120, 130, 140)).save(buf, "JPEG")
    return buf.getvalue()


def _encode_multipart(fields=None, files=None):
    boundary = "zander" + uuid.uuid4().hex
    parts = []
    for name, value in (fields or {}).items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            + value.encode() + b"\r\n"
        )
    for name, (filename, data, ctype) in (files or {}).items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\nContent-Type: {ctype}\r\n\r\n'.encode()
            + data + b"\r\n"
        )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    return body, [(b"content-type", f"multipart/form-data; boundary={boundary}".encode())]


async def _asgi(method: str, path: str, fields=None, files=None, json_body=None):
    """Minimal in-process ASGI client -> (status, json body)."""
    body, headers = (b"", [])
    if fields or files:
        body, headers = _encode_multipart(fields, files)
    elif json_body is not None:
        body, headers = json.dumps(json_body).encode(), [(b"content-type", b"application/json")]
    headers = headers + [(b"host", b"testserver"), (b"content-length", str(len(body)).encode())]
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": method, "scheme": "http",
        "path": path, "raw_path": path.encode(), "root_path": "", "query_string": b"",
        "headers": headers, "client": ("127.0.0.1", 50000), "server": ("testserver", 80),
    }
    sent = False
    done = asyncio.Event()
    out = {"status": None, "body": b""}

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await done.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
        elif message["type"] == "http.response.body":
            out["body"] += message.get("body", b"")
            if not message.get("more_body", False):
                done.set()

    await main.app(scope, receive, send)
    return out["status"], (json.loads(out["body"]) if out["body"] else None)


def _request(method: str, path: str, **kw):
    return asyncio.run(_asgi(method, path, **kw))


def _post_file(data: bytes, filename: str = "photo.jpg", **fields):
    return _request("POST", "/api/v1/inspect", fields=fields or None,
                    files={"file": (filename, data, "image/jpeg")})


# ---- Loading ------------------------------------------------------------------------
def test_no_model_file_is_not_a_crash():
    _reset()
    detector_service.load()
    assert not detector_service.ready
    assert "model not found" in detector_service.load_error
    st = detector_service.status()
    assert set(st) == STATUS_KEYS, set(st) ^ STATUS_KEYS
    assert st["model_loaded"] is False and st["model_version"] is None
    assert set(st["meta"]) == {"split_id", "dataset_fingerprint", "run_name", "created"}


def test_legacy_defaults_without_card():
    _reset()
    _model()
    detector_service.load()
    assert detector_service.ready, detector_service.load_error
    st = detector_service.status()
    sha = sha256_file(MODEL)
    assert st["threshold_source"] == "default" and st["confidence_threshold"] == 0.25
    assert st["adapter"] == "ultralytics" and st["schema"] is None
    assert st["class_names"] == list(DEFAULT_CLASS_NAMES)
    assert st["meta_path"] is None and st["load_error"] is None
    assert st["model_sha256"] == sha and st["model_version"] == f"detector.onnx@{sha[:8]}"
    fake = FakeTiledDetector.instances[-1]
    assert fake.conf_floor == 0.25 and fake.adapter_name == "ultralytics"
    assert any("no detector card" in m for m in LOGS.messages), LOGS.messages


def test_legacy_env_threshold():
    _reset(confidence_threshold=0.4, class_names=["A", "B", "C"])
    _model()
    detector_service.load()
    st = detector_service.status()
    assert st["threshold_source"] == "env" and st["confidence_threshold"] == 0.4
    assert st["class_names"] == ["A", "B", "C"]
    assert FakeTiledDetector.instances[-1].conf_floor == 0.4


def test_legacy_invalid_threshold_is_refused():
    _reset(confidence_threshold=1.5)
    _model()
    detector_service.load()
    assert not detector_service.ready
    assert "CONFIDENCE_THRESHOLD" in detector_service.load_error


def test_card_is_authoritative_and_env_is_ignored():
    _reset(adapter="deim", confidence_threshold=0.9, class_names=["X"])
    _model()
    _card()
    detector_service.load()
    assert detector_service.ready, detector_service.load_error
    st = detector_service.status()
    assert st["threshold_source"] == "detector.json"
    assert st["confidence_threshold"] == 0.34 and st["adapter"] == "ultralytics"
    assert st["class_names"] == NAMES and st["schema"] == "def3"
    assert st["meta_path"] == str(card_path_for(MODEL))
    assert st["meta"] == {
        "split_id": "split_v2", "dataset_fingerprint": "fp123",
        "run_name": "yolo26n_def3_devall", "created": "2026-09-28T10:00:00Z",
    }
    fake = FakeTiledDetector.instances[-1]
    assert fake.conf_floor == 0.34 and fake.adapter_name == "ultralytics"
    warned = [m for m in LOGS.messages if "authoritative" in m]
    assert warned and all(v in warned[0] for v in ("ADAPTER", "CONFIDENCE_THRESHOLD", "CLASS_NAMES"))


def test_card_without_env_overrides_does_not_warn():
    _reset()
    _model()
    _card()
    detector_service.load()
    assert detector_service.ready
    assert not [m for m in LOGS.messages if "authoritative" in m]


def test_sha_mismatch_is_refused_with_503():
    _reset()
    _model()
    _card()
    _model(b"a different model")
    detector_service.load()
    assert not detector_service.ready
    assert "sha256" in detector_service.load_error
    code, body = _post_file(_jpeg())
    assert code == 503, (code, body)
    assert body["detail"].startswith("no model loaded: ") and "sha256" in body["detail"]


def test_class_count_mismatch_is_refused():
    _reset()
    FakeTiledDetector.num_classes = 1
    _model()
    _card()
    detector_service.load()
    assert not detector_service.ready
    assert "class count mismatch" in detector_service.load_error


def test_unknown_class_count_is_accepted():
    _reset()
    FakeTiledDetector.num_classes = None  # e.g. an end-to-end export or DEIM
    _model()
    _card()
    detector_service.load()
    assert detector_service.ready, detector_service.load_error


def test_verifier_card_is_refused():
    _reset()
    _model()
    _card(conf_floor=0.001, verifier={
        "onnx": "verifier.onnx", "onnx_sha256": "0" * 64,
        "stacker": "stacker.json", "stacker_sha256": "0" * 64,
        "top_k": 50, "candidate_floor": 0.001,
    })
    detector_service.load()
    assert not detector_service.ready
    assert "verifier" in detector_service.load_error


def test_require_card_without_card_is_refused():
    _reset(require_card=True)
    _model()
    detector_service.load()
    assert not detector_service.ready
    assert "REQUIRE_CARD" in detector_service.load_error


def test_invalid_card_is_refused():
    _reset()
    _model()
    card_path_for(MODEL).write_text("{not json", encoding="utf-8")
    detector_service.load()
    assert not detector_service.ready
    assert "invalid detector card" in detector_service.load_error


def test_backend_failure_never_raises():
    _reset()
    FakeTiledDetector.raise_on_init = RuntimeError("onnxruntime says no")
    _model()
    _card()
    detector_service.load()  # must not raise
    assert not detector_service.ready
    assert "onnxruntime says no" in detector_service.load_error


def test_reload_after_refusal_recovers():
    _reset()
    _model()
    _card()
    _model(b"tampered")
    detector_service.load()
    assert not detector_service.ready
    _card()  # re-export: card matches the file again
    detector_service.load()
    assert detector_service.ready and detector_service.load_error is None


# ---- Inference -----------------------------------------------------------------------
def test_inspect_response_shape_and_order():
    _reset()
    _model()
    _card()
    detector_service.load()
    r = detector_service.inspect_bytes(_jpeg(700, 500))
    assert r["image_width"] == 700 and r["image_height"] == 500
    assert r["verdict"] == "NiO"
    assert r["classes"] == ["Nicht ausreichend Lot", "Brücke"]  # schema order; Fahne < t
    assert [d["score"] for d in r["detections"]] == [0.9, 0.5]  # highest first
    assert [d["label"] for d in r["detections"]] == ["Brücke", "Nicht ausreichend Lot"]
    assert all(d["score"] >= r["confidence_threshold"] for d in r["detections"])
    assert r["detections"][0] == {"x1": 100.0, "y1": 100.0, "x2": 150.0, "y2": 150.0,
                                  "score": 0.9, "label": "Brücke"}
    assert r["photo_score"] == 0.9 and r["count"] == 2
    assert r["model_sha256"] == sha256_file(MODEL) and r["schema"] == "def3"
    assert r["adapter"] == "ultralytics" and r["class_names"] == NAMES
    assert r["threshold_source"] == "detector.json" and r["tiler_version"] == TILER_VERSION
    assert INSPECT_KEYS - {"status", "source"} <= set(r)


def test_io_verdict_when_nothing_reaches_threshold():
    _reset()
    _model()
    _card(threshold=0.95, conf_floor=0.95)
    detector_service.load()
    r = detector_service.inspect_bytes(_jpeg())
    assert r["verdict"] == "iO" and r["classes"] == [] and r["detections"] == []


def test_undecodable_bytes_raise_decode_error():
    _reset()
    _model()
    _card()
    detector_service.load()
    try:
        detector_service.inspect_bytes(b"definitely not a jpeg")
    except ds.ImageDecodeError:
        pass
    else:
        raise AssertionError("expected ImageDecodeError")


def test_detect_is_serialized_across_threads():
    _reset()
    _model()
    _card()
    detector_service.load()
    jpeg = _jpeg()
    errors: list = []

    def worker():
        try:
            detector_service.inspect_bytes(jpeg)
        except Exception as e:  # noqa: BLE001 (collected for the assert below)
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    assert not errors, errors
    assert FakeTiledDetector.max_active == 1, FakeTiledDetector.max_active
    assert elapsed >= 0.14, f"3 x 50 ms detect ran in {elapsed:.3f}s, so not serialized"


# ---- Startup -------------------------------------------------------------------------
def test_log_level_is_case_insensitive():
    assert main._log_level("debug") == logging.DEBUG
    assert main._log_level(" Warning ") == logging.WARNING
    assert main._log_level("bogus") is None and main._log_level("basic_format") is None
    # The real failure mode: INSPECTOR_LOG_LEVEL=debug -> logging.basicConfig("debug")
    # raised ValueError at import, and Docker restarted the container forever.
    for value, warns in (("debug", False), ("bogus", True)):
        proc = subprocess.run(
            [sys.executable, "-B", "-c", "import main"],
            cwd=TMP, capture_output=True, text=True, timeout=120,
            env={**os.environ, "LOG_LEVEL": value, "PYTHONPATH": str(ROOT / "inspector")},
        )
        assert proc.returncode == 0, (value, proc.stderr[-800:])
        assert ("is not a log level; using INFO" in proc.stderr) is warns, (value, proc.stderr[-800:])


def test_ml_is_appended_to_sys_path_not_prepended():
    # inspector/config and ml/config are both namespace packages with a
    # settings.py: with ml/ in front, a (re-)import of config.settings after
    # detector_service resolved to the ML toolchain's settings.
    ml_dir, inspector_dir = str(ROOT / "ml"), str(ROOT / "inspector")
    assert sys.path.index(ml_dir) > sys.path.index(inspector_dir), sys.path[:4]
    probe = (
        "import sys, services.detector_service; "
        "[sys.modules.pop(m) for m in list(sys.modules) if m == 'config' or m.startswith('config.')]; "
        "import config.settings as s; print(s.__file__)"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", probe], cwd=TMP, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": inspector_dir},
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    assert Path(proc.stdout.strip()) == ROOT / "inspector" / "config" / "settings.py", proc.stdout


# ---- HTTP ----------------------------------------------------------------------------
def test_http_status_and_health_without_model():
    _reset()
    detector_service.load()
    code, st = _request("GET", "/api/v1/status")
    assert code == 200 and st["model_loaded"] is False and "model not found" in st["load_error"]
    assert set(st) == STATUS_KEYS
    code, health = _request("GET", "/api/v1/health")
    assert code == 200 and health == {"status": "ok"}


def test_http_inspect_503_without_model():
    _reset()
    detector_service.load()
    code, body = _post_file(_jpeg())
    assert code == 503 and body["detail"].startswith("no model loaded: model not found"), body


def test_http_inspect_200_has_every_contract_key():
    _reset()
    _model()
    _card()
    detector_service.load()
    code, body = _post_file(_jpeg(800, 600), capture_id="cap-1")
    assert code == 200, (code, body)
    assert set(body) >= INSPECT_KEYS, INSPECT_KEYS - set(body)
    assert body["status"] == "ok" and body["source"] == "cap-1"
    assert body["image_width"] == 800 and body["image_height"] == 600
    assert body["threshold_source"] == "detector.json" and body["confidence_threshold"] == 0.34


def test_http_inspect_422_cases():
    _reset()
    _model()
    _card()
    detector_service.load()
    code, body = _post_file(b"garbage bytes, not an image")
    assert code == 422 and "decode" in body["detail"], (code, body)
    code, body = _post_file(b"")
    assert code == 422, (code, body)
    code, body = _request("POST", "/api/v1/inspect", fields={"capture_id": "../etc"})
    assert code == 422 and body["detail"] == "invalid capture_id", (code, body)
    # fullmatch, not re.match: "$" alone would accept a trailing newline
    code, body = _request("POST", "/api/v1/inspect", fields={"capture_id": "abc\n"})
    assert code == 422 and body["detail"] == "invalid capture_id", (code, body)
    code, body = _post_file(_jpeg(), capture_id="abc\n")  # an upload does not excuse a bad id
    assert code == 422 and body["detail"] == "invalid capture_id", (code, body)
    code, body = _request("POST", "/api/v1/inspect", fields={"other": "x"})
    assert code == 422, (code, body)


def test_http_capture_id_lookup():
    _reset()
    _model()
    _card()
    detector_service.load()
    capture_id = str(uuid.uuid4())
    (DATA / "unlabeled").mkdir(parents=True, exist_ok=True)
    (DATA / "unlabeled" / f"{capture_id}.jpg").write_bytes(_jpeg(900, 700))
    code, body = _request("POST", "/api/v1/inspect", fields={"capture_id": capture_id})
    assert code == 200 and body["source"] == capture_id and body["image_width"] == 900, body
    code, body = _request("POST", "/api/v1/inspect", fields={"capture_id": "unknown-id"})
    assert code == 404, (code, body)


def test_http_inference_error_is_500():
    _reset()
    _model()
    _card()
    detector_service.load()
    saved = FakeTiledDetector.labels
    FakeTiledDetector.labels = [0, 7, 2]  # class index outside the card's 3 names
    try:
        code, body = _post_file(_jpeg())
    finally:
        FakeTiledDetector.labels = saved
    assert code == 500 and body["detail"].startswith("inference failed: "), (code, body)


def test_http_event_loop_stays_responsive_during_inference():
    _reset()
    _model()
    _card()
    detector_service.load()
    FakeTiledDetector.detect_sleep_s = 0.4
    jpeg = _jpeg()

    async def scenario():
        finished = {}

        async def timed(name, coro):
            result = await coro
            finished[name] = time.perf_counter()
            return result

        t0 = time.perf_counter()
        inspect_task = asyncio.create_task(
            timed("inspect", _asgi("POST", "/api/v1/inspect",
                                   files={"file": ("p.jpg", jpeg, "image/jpeg")})))
        await asyncio.sleep(0.05)  # inference is now running in the threadpool
        status = await timed("status", _asgi("GET", "/api/v1/status"))
        inspected = await inspect_task
        return t0, finished, status, inspected

    t0, finished, status, inspected = asyncio.run(scenario())
    assert status[0] == 200 and inspected[0] == 200
    assert finished["status"] < finished["inspect"], "status waited for inference"
    assert finished["status"] - t0 < 0.3, "status blocked behind the 0.4 s detect"


# ---- Real onnxruntime wiring ----------------------------------------------------------------
def _tiny_yolo_onnx(path: Path) -> None:
    """Raw-head layout (1, 4+nc, N) with fixed rows: Brücke 0.9, NAL 0.5, Fahne 0.2."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    n = 16
    base = np.zeros((1, 7, n), dtype=np.float32)
    base[0, :4, 0] = [100, 100, 20, 20]
    base[0, 4 + 1, 0] = 0.9
    base[0, :4, 1] = [300, 300, 40, 40]
    base[0, 4 + 0, 1] = 0.5
    base[0, :4, 2] = [500, 500, 10, 10]
    base[0, 4 + 2, 2] = 0.2
    nodes = [
        helper.make_node("ReduceMax", ["images"], ["m"], keepdims=0),
        helper.make_node("Mul", ["m", "zero"], ["z"]),
        helper.make_node("Add", ["base", "z"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes, "tiny_yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 7, n])],
        initializer=[
            numpy_helper.from_array(base, "base"),
            numpy_helper.from_array(np.array(0, dtype=np.float32), "zero"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_real_onnx_model_end_to_end():
    try:
        import onnx  # noqa: F401
    except ImportError:
        print("SKIP real-onnx case (onnx package missing)")
        return
    _reset()
    ds.TiledDetector = RealTiledDetector
    try:
        _tiny_yolo_onnx(MODEL)
        _card()
        detector_service.load()
        assert detector_service.ready, detector_service.load_error
        r = detector_service.inspect_bytes(_jpeg(640, 640))
        assert r["verdict"] == "NiO"
        assert r["classes"] == ["Nicht ausreichend Lot", "Brücke"]
        assert [(d["label"], d["score"]) for d in r["detections"]] == [
            ("Brücke", 0.9), ("Nicht ausreichend Lot", 0.5)]
        assert r["detections"][0]["x1"] == 90.0 and r["detections"][0]["x2"] == 110.0
        assert r["photo_score"] == 0.9

        # the same model under a 2-class card: the adapter reports 3 -> refused
        _card(class_names=["Nicht ausreichend Lot", "Brücke"])
        detector_service.load()
        assert not detector_service.ready
        assert "class count mismatch" in detector_service.load_error
    finally:
        ds.TiledDetector = FakeTiledDetector


# ---- Several models: the models folder and the active one (CONTRACTS section 3) ----
def _put_active(name=None, **extra):
    body = dict(extra) if name is None else {"name": name, **extra}
    return _request("PUT", "/api/v1/models/active", json_body=body)


def test_models_folder_loads_every_model_and_model_path_is_active():
    _reset(require_card=True)
    _model()
    _card()
    _named_model("fold1")
    _named_model("fold2")
    detector_service.load()
    assert detector_service.ready, detector_service.load_error
    listing = detector_service.models()
    assert set(listing) == MODELS_KEYS, set(listing) ^ MODELS_KEYS
    assert [m["name"] for m in listing["models"]] == ["detector", "fold1", "fold2"]
    assert [m["active"] for m in listing["models"]] == [True, False, False]
    assert all(m["loaded"] and m["error"] is None for m in listing["models"])
    assert listing["active"] == "detector" and listing["selected_by"] is None
    assert listing["models_dir"] == str(MODEL.parent)
    assert detector_service.status()["active_model"] == "detector"
    assert len(FakeTiledDetector.instances) == 3  # one session per model
    assert not settings.active_model_file.exists()  # loading never writes


def test_missing_model_path_falls_back_to_the_first_model_by_name():
    _reset(require_card=True)
    _named_model("zeta")
    _named_model("alpha")
    detector_service.load()
    assert detector_service.ready, detector_service.load_error
    assert detector_service.models()["active"] == "alpha"
    assert any("MODEL_PATH" in m and "alpha" in m for m in LOGS.messages), LOGS.messages


def test_a_refused_model_does_not_stop_the_others():
    _reset(require_card=True)
    _model()
    _card()
    _named_model("nocard", card=False)
    detector_service.load()
    assert detector_service.ready, detector_service.load_error
    entry = {m["name"]: m for m in detector_service.models()["models"]}["nocard"]
    assert entry["loaded"] is False and entry["active"] is False
    assert "REQUIRE_CARD" in entry["error"] and entry["model_version"] is None


def test_a_refused_active_model_is_never_replaced_by_another():
    _reset(require_card=True)
    _model()  # MODEL_PATH without a card: refused
    _named_model("fold1")
    detector_service.load()
    assert not detector_service.ready
    assert "REQUIRE_CARD" in detector_service.load_error
    assert detector_service.models()["active"] == "detector"
    code, body = _post_file(_jpeg())
    assert code == 503 and "REQUIRE_CARD" in body["detail"], body


def test_switch_changes_inspect_and_survives_a_restart():
    _reset(require_card=True)
    _model()
    _card()
    fold1 = _named_model("fold1", threshold=0.6, conf_floor=0.6)
    detector_service.load()
    code, body = _post_file(_jpeg())
    assert code == 200 and body["model_version"].startswith("detector.onnx@"), body
    assert body["classes"] == ["Nicht ausreichend Lot", "Brücke"]  # t 0.34
    code, out = _put_active("fold1", selected_by="op@example.com")
    assert code == 200, out
    assert set(out) == MODELS_KEYS | {"saved"} and out["saved"] is True
    assert out["active"] == "fold1" and out["selected_by"] == "op@example.com"
    assert out["selected_at"].endswith("Z")
    code, body = _post_file(_jpeg())
    assert body["model_version"] == f"fold1.onnx@{sha256_file(fold1)[:8]}", body
    assert body["confidence_threshold"] == 0.6 and body["classes"] == ["Brücke"]  # t 0.6
    saved = json.loads(settings.active_model_file.read_text(encoding="utf-8"))
    assert saved["name"] == "fold1" and saved["model_version"] == body["model_version"]
    assert saved["selected_by"] == "op@example.com"
    detector_service.load()  # a restart keeps the choice
    listing = detector_service.models()
    assert listing["active"] == "fold1" and listing["selected_by"] == "op@example.com"
    assert detector_service.status()["model_version"].startswith("fold1.onnx@")


def test_switch_errors_leave_the_active_model_alone():
    _reset(require_card=True)
    _model()
    _card()
    _named_model("nocard", card=False)
    detector_service.load()
    code, body = _put_active("missing")
    assert code == 404 and "unknown model" in body["detail"], body
    code, body = _put_active("nocard")
    assert code == 409 and "REQUIRE_CARD" in body["detail"], body
    for bad in ({"name": ""}, {}, {"name": 5}):
        code, _ = _request("PUT", "/api/v1/models/active", json_body=bad)
        assert code == 422, bad
    assert detector_service.ready and detector_service.models()["active"] == "detector"
    assert not settings.active_model_file.exists()


def test_a_gone_or_corrupt_saved_choice_falls_back_to_model_path():
    _reset(require_card=True)
    _model()
    _card()
    settings.active_model_file.write_text(json.dumps({"name": "removed"}), encoding="utf-8")
    detector_service.load()
    assert detector_service.ready and detector_service.models()["active"] == "detector"
    assert detector_service.models()["selected_by"] is None
    assert any("saved model choice" in m for m in LOGS.messages), LOGS.messages
    for raw in ("{not json", "[]", '{"name": 5}', '{"name": ""}'):
        settings.active_model_file.write_text(raw, encoding="utf-8")
        detector_service.load()
        assert detector_service.ready, raw
        assert detector_service.models()["active"] == "detector", raw


def test_a_refused_saved_choice_is_loud_and_a_switch_recovers():
    _reset(require_card=True)
    _model()
    _card()
    fold1 = _named_model("fold1")
    detector_service.load()
    assert detector_service.set_active("fold1") is True
    card_path_for(fold1).unlink()  # the chosen model loses its card
    detector_service.load()
    assert not detector_service.ready and "REQUIRE_CARD" in detector_service.load_error
    assert detector_service.models()["active"] == "fold1"
    assert detector_service.set_active("detector") is True  # no restart needed
    assert detector_service.ready and detector_service.load_error is None


def test_an_unsaved_switch_still_applies():
    _reset(require_card=True)
    _model()
    _card()
    _named_model("fold1")
    detector_service.load()
    settings.active_model_file.mkdir()  # os.replace onto a directory fails
    try:
        assert detector_service.set_active("fold1") is False
        assert detector_service.ready and detector_service.models()["active"] == "fold1"
        assert any("not saved" in m for m in LOGS.messages), LOGS.messages
        assert not [p for p in MODEL.parent.iterdir() if p.name.endswith(".tmp")]
    finally:
        settings.active_model_file.rmdir()


def test_the_saved_choice_name_is_reserved():
    _reset()
    _model()
    _card()
    (MODEL.parent / "active_model.onnx").write_bytes(b"fake onnx active_model")
    detector_service.load()
    assert detector_service.ready, detector_service.load_error
    entry = {m["name"]: m for m in detector_service.models()["models"]}["active_model"]
    assert entry["loaded"] is False and "reserved" in entry["error"], entry


def test_models_dir_setting_and_model_path_outside_it():
    other = TMP / "elsewhere"
    shutil.rmtree(other, ignore_errors=True)
    other.mkdir()
    _reset(require_card=True, models_dir=other)
    _model()
    _card()
    _named_model("fold9", folder=other)
    _named_model("detector", folder=other)  # same name as MODEL_PATH: skipped
    detector_service.load()
    assert detector_service.ready, detector_service.load_error
    listing = detector_service.models()
    assert [m["name"] for m in listing["models"]] == ["detector", "fold9"]
    assert listing["active"] == "detector" and listing["models_dir"] == str(other)
    assert detector_service.status()["model_path"] == str(MODEL)
    assert settings.active_model_file.parent == other
    assert any("already loaded" in m for m in LOGS.messages), LOGS.messages


def test_http_models_listing():
    _reset(require_card=True)
    _model()
    _card()
    _named_model("fold1")
    detector_service.load()
    code, body = _request("GET", "/api/v1/models")
    assert code == 200 and set(body) == MODELS_KEYS, body
    first = body["models"][0]
    assert set(first) == MODEL_ENTRY_KEYS, set(first) ^ MODEL_ENTRY_KEYS
    assert first["name"] == "detector" and first["active"] is True
    assert first["run_name"] == "yolo26n_def3_devall" and first["threshold"] == 0.34
    assert first["class_names"] == NAMES and first["threshold_source"] == "detector.json"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    try:
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
    finally:
        os.chdir(ROOT)
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
