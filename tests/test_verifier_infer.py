"""
The torch-free verifier runtime (ml/infer/verifier.py, CONTRACTS §8.4):
crop geometry, seam margin, anchored map, stacker maths, Verifier.rescore on
a tiny constant-logit ONNX graph, stacker.json rejection, and that importing
the module never loads torch.

Run (no pytest needed):  ml/.venv/bin/python -B tests/test_verifier_infer.py
"""
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

import numpy as np  # noqa: E402

import infer.verifier as iv  # noqa: E402

TORCH_FREE_AT_IMPORT = "torch" not in sys.modules

from PIL import Image  # noqa: E402

from infer.pipeline import Detections  # noqa: E402

NAMES = ["Nicht ausreichend Lot", "Brücke", "Fahne"]


def coded_photo(w=700, h=600):
    """Pixel values encode their coordinates (R = x % 256, G = y % 256,
    B = 16 * (x // 256) + y // 256), so any crop can be located exactly."""
    x = np.arange(w)[None, :].repeat(h, 0)
    y = np.arange(h)[:, None].repeat(w, 1)
    return np.stack([x % 256, y % 256, 16 * (x // 256) + y // 256], axis=2).astype(np.uint8)


# ------------------------------------------------------------------ geometry


def test_import_does_not_load_torch():
    assert TORCH_FREE_AT_IMPORT, "importing infer.verifier pulled torch into sys.modules"


def test_window_is_centred_on_the_box():
    photo = coded_photo()
    win = iv.extract_window(photo, (235, 185, 265, 215))  # centre (250, 200), s = 1
    x0, y0 = 250 - 80, 200 - 80
    assert win.shape == (160, 160, 3) and win.dtype == np.uint8
    assert np.array_equal(win, photo[y0:y0 + 160, x0:x0 + 160])
    # the centre pixel of the window is the box centre
    assert tuple(win[80, 80]) == tuple(photo[200, 250])
    # fractional centres round half up: centre 250.5 -> x0 = floor(170.5 + 0.5) = 171
    win2 = iv.extract_window(photo, (235.5, 185, 265.5, 215))
    assert np.array_equal(win2, photo[120:280, 171:331])


def test_reflect_padding_only_on_the_overhang_at_corners():
    photo = coded_photo()
    h, w = photo.shape[:2]
    win = iv.extract_window(photo, (0, 0, 20, 20))  # centre (10, 10): 70 px overhang each side
    want = np.pad(photo[0:90, 0:90], ((70, 0), (70, 0), (0, 0)), mode="reflect")
    assert np.array_equal(win, want)
    # numpy "reflect" mirrors without repeating the edge row/column
    assert tuple(win[70, 70]) == tuple(photo[0, 0])
    assert tuple(win[69, 70]) == tuple(photo[1, 0]) and tuple(win[70, 60]) == tuple(photo[0, 10])
    # the in-bounds part is untouched
    assert np.array_equal(win[70:, 70:], photo[0:90, 0:90])
    # bottom-right corner
    win = iv.extract_window(photo, (w - 20, h - 20, w, h))  # centre (w-10, h-10)
    x0, y0 = w - 10 - 80, h - 10 - 80
    want = np.pad(photo[y0:, x0:], ((0, 70), (0, 70), (0, 0)), mode="reflect")
    assert np.array_equal(win, want)
    assert tuple(win[89, 89]) == tuple(photo[h - 1, w - 1])
    assert tuple(win[90, 89]) == tuple(photo[h - 2, w - 1])


def test_resize_only_when_scale_above_one():
    photo = coded_photo(900, 800)
    # 48-px box: s = 1, native pixels
    assert iv.window_scale((100, 100, 148, 120)) == 1.0
    # 96 x 40 box: s = 2 -> 320-px square resized to 160 with PIL bilinear
    box = (402, 380, 498, 420)  # centre (450, 400)
    assert iv.window_scale(box) == 2.0
    win = iv.extract_window(photo, box)
    square = photo[400 - 160:400 + 160, 450 - 160:450 + 160]
    want = np.asarray(Image.fromarray(square).resize((160, 160), Image.Resampling.BILINEAR))
    assert win.shape == (160, 160, 3) and np.array_equal(win, want)
    # side = round(160 * s): s = 1.25 (60-px box) -> 200-px square
    box = (420, 370, 480, 430)
    square = photo[400 - 100:400 + 100, 450 - 100:450 + 100]
    want = np.asarray(Image.fromarray(square).resize((160, 160), Image.Resampling.BILINEAR))
    assert np.array_equal(iv.extract_window(photo, box), want)


def _reference_window(photo, box):
    """The contract, literally: pad the whole square, then ONE PIL resize."""
    h, w = photo.shape[:2]
    s = max(1.0, max(box[2] - box[0], box[3] - box[1]) / 48)
    side = max(160, int(math.floor(160 * s + 0.5)))
    cx = min(max((box[0] + box[2]) / 2, 0.0), float(w))
    cy = min(max((box[1] + box[3]) / 2, 0.0), float(h))
    x0, y0 = int(math.floor(cx - side / 2 + 0.5)), int(math.floor(cy - side / 2 + 0.5))
    ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
    sq = np.pad(photo[ay0:ay1, ax0:ax1],
                ((ay0 - y0, y0 + side - ay1), (ax0 - x0, x0 + side - ax1), (0, 0)), mode="reflect")
    if side == 160:
        return sq
    return np.asarray(Image.fromarray(sq).resize((160, 160), Image.Resampling.BILINEAR))


def test_large_windows_match_the_literal_contract_bit_for_bit():
    rng = np.random.default_rng(5)
    photo = rng.integers(0, 256, size=(600, 800, 3), dtype=np.uint8)
    boxes = [
        (10, 10, 70, 50),        # s = 1.25, top-left overhang on both axes
        (700, 520, 796, 598),    # s = 2, bottom-right overhang
        (300, 5, 400, 40),       # top overhang only
        (5, 250, 60, 300),       # left overhang only
        (100, 100, 700, 500),    # s = 12.5: the square is larger than the photo
        (350, 250, 450, 350),    # s > 1 without any overhang
        (-40, -30, 900, 650),    # box bigger than the photo, centre clamped
    ]
    for box in boxes:
        assert np.array_equal(iv.extract_window(photo, box), _reference_window(photo, box)), box


def test_crop_offsets_and_shared_shift():
    win = coded_photo(160, 160)
    small, large = iv.crops_from_window(win)
    assert large.shape == (128, 128, 3) and small.shape == (64, 64, 3)
    assert np.array_equal(large, win[16:144, 16:144]) and np.array_equal(small, win[48:112, 48:112])
    small, large = iv.crops_from_window(win, dx=3, dy=-5)
    assert np.array_equal(large, win[11:139, 19:147]) and np.array_equal(small, win[43:107, 51:115])
    # both crops move together: small stays the exact centre of large
    assert np.array_equal(large[32:96, 32:96], small)
    for bad in ((9, 0), (0, -9)):
        try:
            iv.crops_from_window(win, *bad)
            raise AssertionError("a shift beyond the jitter margin must be refused")
        except ValueError:
            pass
    feed = iv.batch_inputs(np.stack([win, win[::-1].copy()]))
    assert feed["crop_small"].shape == (2, 3, 64, 64) and feed["crop_large"].shape == (2, 3, 128, 128)
    assert feed["crop_small"].dtype == np.float32
    assert np.allclose(feed["crop_large"][0], win[16:144, 16:144].transpose(2, 0, 1) / 255.0)


def test_seam_margin_hand_computed():
    W, H = 2616, 1960  # x origins {0,512,1024,1536,1976}, y origins {0,512,1024,1320}
    # 4 tiles contain it; tile (512,512): left 88, right 522, top 88, bottom 522 -> 88
    assert iv.seam_margin((600, 600, 630, 630), W, H) == 88 / 128
    # straddles the x=640 seam of tile (0,0): only tile (512,0): left 118, bottom 510 -> 118
    assert iv.seam_margin((630, 100, 660, 130), W, H) == 118 / 128
    # only tile (1024,512): right 1664-1530 = 134, bottom 1152-1030 = 122 -> 122
    assert iv.seam_margin((1500, 1000, 1530, 1030), W, H) == 122 / 128
    # top-left photo corner: the photo border is no seam -> margin 600, capped at 128
    assert iv.seam_margin((10, 10, 40, 40), W, H) == 1.0
    # boxes sticking out of the photo are clipped first
    assert iv.seam_margin((-5, 10, 25, 40), W, H) == 1.0
    # wider than a tile: no tile contains it
    assert iv.seam_margin((400, 100, 1200, 130), W, H) == 0.0
    # a photo smaller than one tile has no seams at all
    assert iv.seam_margin((100, 100, 130, 130), 500, 400) == 1.0


# ------------------------------------------------------------------ scores


def test_anchor_map_monotone_and_inclusive():
    tau, t = 0.37, 0.28
    assert iv.anchor_map(tau, tau, t) == t
    assert iv.anchor_map(0.0, tau, t) == 0.0 and iv.anchor_map(1.0, tau, t) == 1.0
    assert isinstance(iv.anchor_map(0.5, tau, t), float)
    grid = np.linspace(0, 1, 2001)
    a = iv.anchor_map(grid, tau, t)
    assert np.all(np.diff(a) >= 0)
    rng = np.random.default_rng(0)
    ps = np.concatenate([rng.random(5000), [tau, np.nextafter(tau, 0), np.nextafter(tau, 1), 0, 1]])
    for p in ps:
        assert (iv.anchor_map(p, tau, t) >= t) == (p >= tau), p
    # hand values: below tau linear to t*, above tau linear to 1
    assert math.isclose(iv.anchor_map(tau / 2, tau, t), t / 2)
    assert math.isclose(iv.anchor_map((1 + tau) / 2, tau, t), (1 + t) / 2)
    for bad in ((0.5, 0.0, 0.3), (0.5, 1.0, 0.3), (0.5, 0.5, 1.0)):
        try:
            iv.anchor_map(*bad)
            raise AssertionError("tau and t* must be in (0, 1)")
        except ValueError:
            pass


def stacker_doc(onnx_sha, coef=(1.0, 2.0, 0.0, 0.0), mean=(0, 0, 0, 0), std=(1, 1, 1, 1),
                intercept=-1.0, tau=0.5, t_star=0.3, hi=None, top_k=50, names=NAMES):
    classes = {n: {"mode": "passthrough"} for n in names}
    classes[names[0]] = {"mode": "stack", "mean": list(mean), "std": list(std),
                         "coef": list(coef), "intercept": intercept, "tau": tau,
                         "hi_passthrough": hi}
    return {"contract_version": 1, "kind": "zander-stacker", "schema": "def3",
            "class_names": list(names), "verifier_file": "verifier.onnx",
            "verifier_sha256": onnx_sha, "top_k": top_k, "candidate_floor": 0.001,
            "detector_threshold": t_star,
            "crop": {"window": 160, "small": 64, "large": 128, "scale_ref": 48, "jitter": 8},
            "features": ["det_logit", "ver_logit", "log_area", "seam_margin"],
            "classes": classes, "anchor": "piecewise_linear", "knn": None,
            "provenance": {}, "exit": None}


def test_stacker_maths_with_hand_weights():
    st = iv.Stacker.from_dict(stacker_doc("0" * 64))
    box = (600, 600, 630, 630)
    x = st.features([0.5], [1.0], [box], 2616, 1960)
    assert np.allclose(x[0], [0.0, 1.0, math.log(900.0), 88 / 128])
    # z = -1 + 1*det_logit(0) + 2*ver_logit(1) = 1
    p = st.proba(0, x)
    assert math.isclose(p[0], 1 / (1 + math.exp(-1)), rel_tol=1e-12)
    final = st.final(0, p, [0.5])
    assert math.isclose(final[0], 0.3 + 0.7 * (p[0] - 0.5) / 0.5, rel_tol=1e-12)
    # standardisation: (ver_logit - 1) / 2 = 0 -> z = -1 -> p = 0.2689 < tau -> t* * p / tau
    st2 = iv.Stacker.from_dict(stacker_doc("0" * 64, mean=(0, 1, 0, 0), std=(1, 2, 1, 1)))
    p2 = st2.proba(0, x)
    assert math.isclose(p2[0], 1 / (1 + math.exp(1)), rel_tol=1e-12)
    assert math.isclose(st2.final(0, p2, [0.5])[0], 0.3 * p2[0] / 0.5, rel_tol=1e-12)
    # hi-passthrough: a detection at 0.95 >= h = 0.9 keeps at least its detector score
    st3 = iv.Stacker.from_dict(stacker_doc("0" * 64, hi=0.9))
    assert st3.final(0, [0.01], [0.95])[0] == 0.95
    assert st3.final(0, [0.01], [0.85])[0] < 0.3
    # passthrough classes return the detector score unchanged
    assert list(st.final(1, [0.99], [0.123])) == [0.123]
    assert iv.logit(0.5) == 0.0 and math.isclose(iv.logit(0.0), math.log(1e-6 / (1 - 1e-6)))


# ------------------------------------------------------------------ runtime


def constant_logit_onnx(path: Path, logits=(2.0, -1.0, 0.5)):
    """crop_small/crop_large -> logits [N,C] = const, embedding [N,128] = 0 (opset 17)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    c = len(logits)
    nodes = [
        helper.make_node("ReduceMean", ["crop_small"], ["ms"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("ReduceMean", ["crop_large"], ["ml"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Add", ["ms", "ml"], ["m"]),
        helper.make_node("Unsqueeze", ["m", "ax1"], ["m1"]),
        helper.make_node("Mul", ["m1", "zero"], ["z"]),
        helper.make_node("Add", ["z", "lconst"], ["logits"]),
        helper.make_node("Add", ["z", "econst"], ["embedding"]),
    ]
    inits = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), "ax1"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero"),
        numpy_helper.from_array(np.array([logits], dtype=np.float32), "lconst"),
        numpy_helper.from_array(np.zeros((1, 128), dtype=np.float32), "econst"),
    ]
    graph = helper.make_graph(
        nodes, "const_verifier",
        [helper.make_tensor_value_info("crop_small", TensorProto.FLOAT, ["N", 3, 64, 64]),
         helper.make_tensor_value_info("crop_large", TensorProto.FLOAT, ["N", 3, 128, 128])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", c]),
         helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["N", 128])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_pair(d: Path, doc: dict) -> Path:
    p = d / "stacker.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_rescore_topk_passthrough_and_order():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        sha = constant_logit_onnx(d / "verifier.onnx")
        # NAL: p = sigmoid(ver_logit) = sigmoid(2) -> A = 0.4 + 0.6 (p - 0.5) / 0.5
        doc = stacker_doc(sha, coef=(0, 1, 0, 0), intercept=0.0, tau=0.5, t_star=0.4, top_k=3)
        ver = iv.Verifier(write_pair(d, doc), expected_class_names=NAMES)
        photo = coded_photo(900, 700)
        boxes = [(100, 100, 130, 130), (200, 200, 230, 230), (300, 300, 330, 330),
                 (400, 400, 430, 430), (500, 500, 530, 530)]
        scores = [0.9, 0.1, 0.5, 0.05, 0.2]
        labels = [1, 0, 0, 2, 0]
        out = ver.rescore(photo, Detections(boxes, scores, labels, 12.5))
        p = 1 / (1 + math.exp(-2.0))
        a = 0.4 + 0.6 * (p - 0.5) / 0.5
        assert len(out.scores) == 3  # top-k by detector score; 0.1 NAL and 0.05 Fahne dropped
        assert out.labels == [1, 0, 0]
        assert out.boxes == [boxes[0], boxes[2], boxes[4]]
        assert out.scores[0] == 0.9  # passthrough class keeps its detector score
        assert math.isclose(out.scores[1], a, rel_tol=1e-6) and math.isclose(out.scores[2], a, rel_tol=1e-6)
        assert out.scores == sorted(out.scores, reverse=True)
        assert out.elapsed_ms >= 12.5
        # the verifier now rejects NAL: its scores drop below t*, Brücke untouched
        doc2 = stacker_doc(sha, coef=(0, -1, 0, 0), intercept=0.0, tau=0.5, t_star=0.4, top_k=3)
        out2 = iv.Verifier(write_pair(d, doc2)).rescore(photo, Detections(boxes, scores, labels, 0.0))
        low = 0.4 * (1 / (1 + math.exp(2.0))) / 0.5
        assert out2.labels == [1, 0, 0] and out2.scores[0] == 0.9
        assert math.isclose(out2.scores[1], low, rel_tol=1e-6) and out2.scores[1] < 0.4
        # a stronger NAL score can overtake a passthrough detection
        doc3 = stacker_doc(sha, coef=(0, 5, 0, 0), intercept=0.0, tau=0.5, t_star=0.4, top_k=5)
        out3 = iv.Verifier(write_pair(d, doc3)).rescore(photo, Detections(boxes, scores, labels, 0.0))
        assert out3.labels[:3] == [0, 0, 0] and out3.labels[3:] == [1, 2]
        assert out3.scores == sorted(out3.scores, reverse=True)
        # no detections at all
        empty = ver.rescore(photo, Detections([], [], [], 1.0))
        assert empty.boxes == [] and empty.scores == [] and empty.labels == []


def test_stacker_json_mismatches_are_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        sha = constant_logit_onnx(d / "verifier.onnx")

        def refused(doc, **kw):
            try:
                iv.Verifier(write_pair(d, doc), **kw)
            except iv.StackerError:
                return True
            return False

        assert not refused(stacker_doc(sha))
        assert refused(stacker_doc("f" * 64))  # sha of another model
        assert refused(stacker_doc(sha, names=NAMES[:2]))  # 3 heads in the graph, 2 classes
        assert refused(stacker_doc(sha), expected_class_names=["Nicht ausreichend Lot"])
        assert refused(stacker_doc(sha, tau=1.5))
        bad = stacker_doc(sha)
        bad["crop"]["window"] = 192
        assert refused(bad)
        bad = stacker_doc(sha)
        bad["features"] = ["det_logit", "ver_logit"]
        assert refused(bad)
        bad = stacker_doc(sha)
        bad["classes"]["Nicht ausreichend Lot"]["std"] = [1, 0, 1, 1]
        assert refused(bad)
        bad = stacker_doc(sha)
        bad["exit"] = {"decision": "maybe"}
        assert refused(bad)
        ok = stacker_doc(sha)
        ok["exit"] = {"decision": "invalid-smoke", "report": "exit_report.md"}
        assert not refused(ok)


def test_still_torch_free_after_all_tests():
    assert "torch" not in sys.modules


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
