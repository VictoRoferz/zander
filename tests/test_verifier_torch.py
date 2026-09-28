"""
The torch side of the verifier (ml/verifier/{model,train,export_onnx}.py):
network output shapes, ONNX export parity (<= 1e-4, dynamic batch), the
training crops reuse the inference geometry, D4 follows infer.tiling's
convention, a tiny CPU train/predict, and torch -> ONNX -> torch-free Verifier.
Skipped entirely when torch is not installed (the inspector image).

Run (no pytest needed):  ml/.venv/bin/python -B tests/test_verifier_torch.py
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    print("SKIP: torch is not installed")
    sys.exit(0)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from infer.pipeline import Detections  # noqa: E402
from infer.tiling import d4_image  # noqa: E402
from infer.verifier import Verifier, batch_inputs, crops_from_window  # noqa: E402
from verifier.config import load_config  # noqa: E402
from verifier.export_onnx import export_model, parity  # noqa: E402
from verifier.model import TwoCropVerifier  # noqa: E402
from verifier.train import augment_batch, d4_np, jittered_crops, predict, train_model  # noqa: E402

torch.manual_seed(0)
RNG = np.random.default_rng(0)
WINDOWS = RNG.integers(0, 256, size=(24, 160, 160, 3), dtype=np.uint8)
NAMES = ["Nicht ausreichend Lot", "Brücke", "Fahne"]


def tiny_model(c=3):
    m = TwoCropVerifier(c, pretrained=False)
    m.eval()
    return m


def test_output_shapes():
    m = tiny_model()
    assert m.weights_source == "none"
    for n in (1, 5):
        feed = batch_inputs(WINDOWS[:n])
        with torch.no_grad():
            logits, emb = m(torch.from_numpy(feed["crop_small"]), torch.from_numpy(feed["crop_large"]))
        assert tuple(logits.shape) == (n, 3) and tuple(emb.shape) == (n, 128)
    assert m.fc.in_features == 1152  # 576 pooled features per crop, concatenated


def test_onnx_export_parity_and_dynamic_batch():
    m = tiny_model()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "verifier.onnx"
        feed = batch_inputs(WINDOWS[:2])
        export_model(m, path, torch.from_numpy(feed["crop_small"]), torch.from_numpy(feed["crop_large"]))
        par = parity(m, path, WINDOWS)  # 24 windows through a graph exported at N = 2
        assert par["ok"], par
        assert par["max_abs_logits"] <= 1e-4 and par["max_abs_embedding"] <= 1e-4
        assert par["logits_shape"] == [24, 3] and par["embedding_shape"] == [24, 128]
        import onnx

        model = onnx.load(str(path))
        assert model.opset_import[0].version == 17
        assert [i.name for i in model.graph.input] == ["crop_small", "crop_large"]
        assert [o.name for o in model.graph.output] == ["logits", "embedding"]


def test_training_crops_reuse_the_inference_geometry():
    dx, dy = RNG.integers(-8, 9, 24), RNG.integers(-8, 9, 24)
    small, large = jittered_crops(WINDOWS, dx, dy)
    for i in range(24):
        s, l = crops_from_window(WINDOWS[i], int(dx[i]), int(dy[i]))
        assert np.array_equal(small[i], s) and np.array_equal(large[i], l)
    # zero shift = exactly what the ONNX path is fed
    small0, large0 = jittered_crops(WINDOWS, np.zeros(24, int), np.zeros(24, int))
    feed = batch_inputs(WINDOWS)
    assert np.allclose(small0.transpose(0, 3, 1, 2) / 255.0, feed["crop_small"])


def test_d4_follows_the_tiler_convention():
    img = RNG.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    for k in range(8):
        want = np.asarray(d4_image(Image.fromarray(img), k))
        assert np.array_equal(d4_np(img, k), want), k


def test_augmentation_identity_and_ranges():
    ident = load_config(overrides={"augment.jitter": 0, "augment.d4": False, "augment.brightness": 0,
                                   "augment.contrast": 0, "augment.saturation": 0, "augment.hue": 0})
    small, large = augment_batch(WINDOWS[:6], ident, np.random.default_rng(1), torch.device("cpu"))
    feed = batch_inputs(WINDOWS[:6])
    assert np.allclose(small.numpy(), feed["crop_small"], atol=1e-6)
    assert np.allclose(large.numpy(), feed["crop_large"], atol=1e-6)
    small, large = augment_batch(WINDOWS[:6], load_config(), np.random.default_rng(2), torch.device("cpu"))
    assert tuple(small.shape) == (6, 3, 64, 64) and tuple(large.shape) == (6, 3, 128, 128)
    assert float(small.min()) >= 0.0 and float(large.max()) <= 1.0


def test_tiny_train_and_predict_on_cpu():
    cfg = load_config(overrides={"model.pretrained": "none", "train.epochs": 2, "train.batch_size": 8})
    ids = np.arange(24)
    labels = np.array([1, 0, 0] * 8)
    classes = np.zeros(24, dtype=int)
    model, log = train_model("tiny", WINDOWS, ids, labels, classes, 3, cfg, torch.device("cpu"), seed=7)
    assert log["n_train"] == 24 and log["n_pos"] == 8 and len(log["epoch_log"]) == 2
    assert all(np.isfinite(e["loss"]) for e in log["epoch_log"])
    z = predict(model, WINDOWS, ids[::-1], classes, torch.device("cpu"), batch_size=5)
    z2 = predict(model, WINDOWS, ids, classes, torch.device("cpu"), batch_size=24)
    assert z.shape == (24,) and np.all(np.isfinite(z))
    assert np.allclose(z[::-1], z2, atol=1e-5)  # order-preserving, batch-size independent
    # the same seed reproduces the same model on CPU
    model_b, _ = train_model("tiny", WINDOWS, ids, labels, classes, 3, cfg, torch.device("cpu"), seed=7)
    assert np.allclose(predict(model_b, WINDOWS, ids, classes, torch.device("cpu")), z2, atol=1e-5)


def test_torch_to_onnx_to_torch_free_verifier():
    m = tiny_model()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        feed = batch_inputs(WINDOWS[:2])
        export_model(m, d / "verifier.onnx", torch.from_numpy(feed["crop_small"]),
                     torch.from_numpy(feed["crop_large"]))
        sha = hashlib.sha256((d / "verifier.onnx").read_bytes()).hexdigest()
        classes = {n: {"mode": "passthrough"} for n in NAMES}
        classes[NAMES[0]] = {"mode": "stack", "mean": [0, 0, 0, 0], "std": [1, 1, 1, 1],
                             "coef": [0.5, 1.0, 0.0, 0.0], "intercept": 0.0, "tau": 0.4,
                             "hi_passthrough": None}
        doc = {"contract_version": 1, "kind": "zander-stacker", "schema": "def3",
               "class_names": NAMES, "verifier_file": "verifier.onnx", "verifier_sha256": sha,
               "top_k": 50, "candidate_floor": 0.001, "detector_threshold": 0.3,
               "crop": {"window": 160, "small": 64, "large": 128, "scale_ref": 48, "jitter": 8},
               "features": ["det_logit", "ver_logit", "log_area", "seam_margin"],
               "classes": classes, "anchor": "piecewise_linear", "knn": None,
               "provenance": {}, "exit": None}
        (d / "stacker.json").write_text(json.dumps(doc), encoding="utf-8")
        ver = Verifier(d / "stacker.json", expected_class_names=NAMES)
        photo = RNG.integers(0, 256, size=(700, 900, 3), dtype=np.uint8)
        boxes = [(100, 100, 130, 130), (400, 300, 440, 330), (800, 650, 900, 700)]
        det = Detections(boxes, [0.2, 0.6, 0.05], [0, 1, 0], 0.0)
        r = ver.score_candidates(photo, det.boxes, det.scores, det.labels)
        # the runtime's verifier logits equal torch's on the same windows
        wins = np.stack([ver.windows(photo, [boxes[0]])[0], ver.windows(photo, [boxes[2]])[0]])
        f = batch_inputs(wins)
        with torch.no_grad():
            lt, _ = m(torch.from_numpy(f["crop_small"]), torch.from_numpy(f["crop_large"]))
        got = r["ver_logit"][[list(r["order"]).index(0), list(r["order"]).index(2)]]
        assert np.allclose(got, lt[:, 0].numpy(), atol=1e-4)
        out = ver.rescore(photo, det)
        assert out.scores == sorted(out.scores, reverse=True) and 0.6 in out.scores


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
