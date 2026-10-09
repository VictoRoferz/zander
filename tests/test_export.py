"""
Detector export (ml/export.py -> detector.onnx + detector.json, CONTRACTS §1).

A tiny onnx.helper graph returns a CONSTANT raw Ultralytics head (1, 7, N)
for every tile, so the full tiled pipeline runs on a synthetic 2616x1960
photo with a known verdict (t=0.5: N.iO, classes NAL + Brücke). Covers the
card round trip through infer/detector_card, the refusals (fold model, tiler,
schema, config, stale operating point, missing test eval, v1 operating point,
existing export), the choice of the test look recorded at this op, sha
tampering, the parity check (reference mode, live floor mode, a forced
mismatch, the v1 rounding tolerance) and the floor-equivalence property that
makes conf_floor = threshold safe.

Run: ml/.venv/bin/python -B tests/test_export.py
"""
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="zander_export_"))
os.environ["ML_DATA_ROOT"] = str(TMP)  # BEFORE config.settings is imported
sys.path.insert(0, str(ROOT / "ml"))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402
from PIL import Image  # noqa: E402

import export  # noqa: E402
import registry  # noqa: E402
from config.settings import settings  # noqa: E402
from dataprep.split import load_split  # noqa: E402
from infer.detector_card import DetectorCardError, load_detector_card  # noqa: E402
from infer.pipeline import TiledDetector  # noqa: E402
from infer.stitch import merge_by_class  # noqa: E402
from infer.verdict import photo_score, photo_verdict  # noqa: E402

NAMES = ["Nicht ausreichend Lot", "Brücke", "Fahne"]
SPLIT_ID = load_split()["split_id"]
T = 0.5
# tile-pixel rows: cx, cy, w, h, score NAL, Brücke, Fahne
ROWS = [
    (100, 100, 20, 20, 0.9, 0.05, 0.01),
    (300, 300, 30, 30, 0.02, 0.6, 0.03),
    (500, 100, 16, 16, 0.01, 0.02, 0.35),
    (200, 500, 24, 24, 0.2, 0.1, 0.0),
    (400, 550, 20, 20, 0.004, 0.003, 0.002),
    (560, 560, 20, 20, 0.0005, 0.0004, 0.0003),
    (50, 600, 10, 10, 0.3, 0.31, 0.29),
    (610, 30, 20, 20, 0.45, 0.1, 0.1),
]
PHOTO_IDS = ["synth0001", "synth0002", "synth0003"]
PHOTOS: dict = {}


def _tiny_onnx(path: Path, rows=ROWS) -> Path:
    raw = np.asarray(rows, dtype=np.float32).T[None]  # (1, 7, N)
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["m"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Mul", ["m", "zero"], ["z"]),
        helper.make_node("Add", ["raw", "z"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes, "tiny_constant_head",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(raw.shape))],
        initializer=[numpy_helper.from_array(raw, "raw"),
                     numpy_helper.from_array(np.zeros((1,), np.float32), "zero")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def _bundle(run: str, fold: str, with_model=True, rows=ROWS, **meta_over) -> Path:
    d = settings.bundles_dir / run
    d.mkdir(parents=True, exist_ok=True)
    if with_model:
        _tiny_onnx(d / "model.onnx", rows=rows)
    meta = {"bundle_version": 2, "run_name": run, "family": "tiny_def3", "fold": fold,
            "framework": "ultralytics", "adapter": "ultralytics", "schema": "def3",
            "classes": NAMES, "split_id": SPLIT_ID, "tiler_version": "v2",
            "dataset_id": "ds_v2_def3", "dataset_fingerprint": "f" * 64, "snapshot_id": "s",
            "weights": "last", "git_sha": "abc123", "versions": {"ultralytics": "8.4.115"},
            "onnx": {"file": "model.onnx", "layout": "raw", "opset": 13}}
    epochs = meta_over.pop("epochs", 120)
    meta.update(meta_over)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "config.yaml").write_text(
        f"name: {run}\nmodel: tiny.pt\nadapter: ultralytics\ndataset_id: ds_v2_def3\nfold: {fold}\n"
        f"epochs: {epochs}\nimgsz: 640\nbatch: 16\nfinal_weights: last\n", encoding="utf-8")
    return d


def _fold_model_sha(i: int) -> str:
    """The model choose-op froze the op from: fold i's model.onnx (only fold0 has
    one; a bundle without model and onnx.sha256 cannot be checked for staleness)."""
    model = settings.bundles_dir / f"tiny_def3_fold{i}" / "model.onnx"
    return registry.file_sha256(model) if model.is_file() else str(i) * 64


def _op(name="operating_point_def3_tiny_def3.json", **over) -> Path:
    op = {"schema": "def3", "class_names": NAMES, "threshold": T,
          "rule": "highest t with pooled out-of-fold photo sensitivity >= 0.95",
          "runs": [f"tiny_def3_fold{i}" for i in range(3)], "models": [], "split_id": SPLIT_ID,
          "created": "2026-09-28T00:00:00Z", "oof_level1": {"n_positive": 3},
          "family": "tiny_def3", "target_sensitivity": 0.95,
          "sources": [{"eval_run": f"tiny_def3_fold{i}", "bundle_run": f"tiny_def3_fold{i}",
                       "fold": f"fold{i}", "model_sha256": _fold_model_sha(i), "config_sha256": None,
                       "dataset_fingerprint": "f" * 64} for i in range(3)]}
    op.update(over)
    for key in [k for k, v in over.items() if v is None]:
        del op[key]
    path = settings.eval_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(op, ensure_ascii=False), encoding="utf-8")
    return path


def _eval_run(name: str, bundle: Path, bucket: str, op: Path | None, fmt=2, tamper=None):
    """An eval run of the bundle's model: predictions computed by the real pipeline."""
    model = bundle / "model.onnx"
    det = TiledDetector(model, "ultralytics", conf_floor=0.001)
    photos = []
    for cid in PHOTO_IDS:
        d = det.detect_file(PHOTOS[cid])
        p = {"capture_id": cid, "group": cid, "board": "A", "pose": "seated", "gt_verdict": "NiO",
             "gt": [], "other_gt": [], "boxes": [[round(v, 1) for v in b] for b in d.boxes],
             "scores": [float(s) for s in d.scores], "labels": list(d.labels)}
        if fmt == 2:
            p["width"], p["height"] = 2616, 1960
        photos.append(p)
    if tamper:
        tamper(photos)
    doc = {"schema": "def3", "class_names": NAMES, "bucket": bucket, "model": str(model), "photos": photos}
    sha = registry.file_sha256(model)
    metrics = {"run_name": name, "bucket": bucket, "threshold": T if op else None,
               "level1_verdict": {"n_positive": 3, "escape": {"k": 0, "n": 3, "rate": 0.0}},
               "model": str(model), "mean_ms_per_photo": 1.0, "n_photos": 3}
    if fmt == 2:
        doc["format_version"] = 2
        doc["source"] = metrics["source"] = {"kind": "detector", "model_sha256": sha,
                                             "bundle_run": bundle.name, "fold": "dev_all"}
        if op:
            metrics["operating_point"] = {"file": str(op), "sha256": registry.file_sha256(op),
                                          "threshold": T}
    d = settings.eval_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "predictions.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return d


def _raises(fn, exc, *needles):
    try:
        fn()
    except exc as e:
        for n in needles:
            assert n in str(e), f"{n!r} not in: {e}"
        return e
    raise AssertionError(f"expected {exc.__name__}")


# ---- fixture: photos, bundles, operating point, test eval ----
for cid in PHOTO_IDS:
    PHOTOS[cid] = TMP / "photos" / f"{cid}.png"
    PHOTOS[cid].parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2616, 1960), (90, 110, 70)).save(PHOTOS[cid])
export._photo_paths = lambda ids: {c: PHOTOS[c] for c in ids if c in PHOTOS}
DEVALL = _bundle("tiny_def3_devall", "dev_all")
FOLD0_ROWS = ROWS[:-1] + [(610, 30, 20, 20, 0.46, 0.1, 0.1)]  # its own model file (own sha)
for i in range(3):
    _bundle(f"tiny_def3_fold{i}", f"fold{i}", with_model=(i == 0), rows=FOLD0_ROWS)
OP = _op()
_eval_run("tiny_def3_devall__test", DEVALL, "test", OP)
OUT = TMP / "exports"


def test_known_verdict_of_the_tiny_model():
    det = TiledDetector(DEVALL / "model.onnx", "ultralytics", conf_floor=T).detect_file(PHOTOS["synth0001"])
    v = photo_verdict(det.boxes, det.scores, det.labels, T, NAMES)
    assert v["verdict"] == "NiO" and v["classes"] == ["Nicht ausreichend Lot", "Brücke"], v
    assert len(det.boxes) == 40  # 20 tiles x (NAL 0.9 + Brücke 0.6), no cross-tile merges


def test_export_writes_a_valid_card():
    card_path = export.run(DEVALL, OP, out_dir=OUT)
    assert card_path == OUT / "detector.json" and (OUT / "detector.onnx").is_file()
    card = load_detector_card(OUT / "detector.onnx")  # the inspector's reader
    assert card.threshold == card.conf_floor == T and card.adapter == "ultralytics"
    assert card.class_names == tuple(NAMES) and card.schema == "def3" and card.verifier is None
    raw = card.raw
    assert raw["model_file"] == "detector.onnx"
    assert raw["model_sha256"] == registry.file_sha256(DEVALL / "model.onnx")
    assert raw["source"]["bundle_run"] == "tiny_def3_devall" and raw["source"]["fold"] == "dev_all"
    assert raw["operating_point"]["family"] == "tiny_def3"
    assert raw["operating_point"]["sha256"] == registry.file_sha256(OP)
    val = raw["validation"]
    assert val["test_eval_run"] == "tiny_def3_devall__test" and val["overrides"] == []
    assert val["test_level1"]["n_positive"] == 3
    assert val["parity"]["verdict_mismatches"] == 0 and val["parity"]["photos"] == 3, val["parity"]
    assert val["parity"]["mode"] == "reference" and val["parity"]["score_mismatches"] == 0
    assert raw["exported_by"].endswith("cli export") and raw["exported_at"].endswith("Z")
    assert not [p for p in OUT.iterdir() if p.name.startswith(".")], "staging left behind"
    assert oct((OUT / "detector.onnx").stat().st_mode & 0o777) == "0o644"


def test_existing_export_needs_force_and_is_archived():
    _raises(lambda: export.run(DEVALL, OP, out_dir=OUT), export.ExportError, "--force")
    export.run(DEVALL, OP, out_dir=OUT, force=True)
    archived = list((OUT / "archive").iterdir())
    assert len(archived) == 1 and {p.name for p in archived[0].iterdir()} == {"detector.onnx", "detector.json"}
    load_detector_card(OUT / "detector.onnx")


def test_sha_and_tiler_tampering_rejected():
    out = TMP / "exports_tamper"
    export.run(DEVALL, OP, out_dir=out)
    (out / "detector.onnx").write_bytes((out / "detector.onnx").read_bytes() + b"\0")
    _raises(lambda: load_detector_card(out / "detector.onnx"), DetectorCardError, "sha256 mismatch")
    card = json.loads((out / "detector.json").read_text(encoding="utf-8"))
    card["tiler_version"] = "v1"
    (out / "detector.json").write_text(json.dumps(card), encoding="utf-8")
    _raises(lambda: load_detector_card(out / "detector.onnx", verify_sha=False), DetectorCardError, "tiler")
    old_tiler = _bundle("tiny_def3_devall_t1", "dev_all", tiler_version="v1", family="tiny_def3")
    _eval_run("tiny_def3_devall_t1__test", old_tiler, "test", OP)
    _raises(lambda: export.run(old_tiler, OP, out_dir=TMP / "x1"), export.ExportError, "tiler")


def test_refusals():
    fold0 = settings.bundles_dir / "tiny_def3_fold0"
    _raises(lambda: export.run(fold0, OP, out_dir=TMP / "x2"), export.ExportError, "dev_all", "--allow-fold-model")
    nal_op = _op("op_nal1.json", schema="nal1", class_names=NAMES[:1])
    _raises(lambda: export.run(DEVALL, nal_op, out_dir=TMP / "x3"), export.ExportError, "schema")
    other_fam = _op("op_other_family.json", family="other_def3")
    _raises(lambda: export.run(DEVALL, other_fam, out_dir=TMP / "x4"), export.ExportError, "family")
    _bundle("tiny_def3_fold2", "fold2", with_model=False, epochs=2)  # recipe drift in one fold
    try:
        _raises(lambda: export.run(DEVALL, OP, out_dir=TMP / "x5"), export.ExportError, "epochs differs")
    finally:
        _bundle("tiny_def3_fold2", "fold2", with_model=False)
    missing = _op("op_missing_sources.json", sources=[{"bundle_run": "never_imported_fold0"}])
    _raises(lambda: export.run(DEVALL, missing, out_dir=TMP / "x6"), export.ExportError, "not imported")
    v1 = _op("op_v1.json", family=None, sources=None, target_sensitivity=None)
    _raises(lambda: export.run(DEVALL, v1, out_dir=TMP / "x7"), export.ExportError, "v1 operating point")
    _raises(lambda: export.run(DEVALL, OP, out_dir=TMP / "x8", name="../evil"), export.ExportError, "--name")


def test_test_eval_required_unless_waived():
    fresh = _bundle("tiny2_def3_devall", "dev_all", family="tiny_def3",
                    onnx={"file": "model.onnx", "layout": "raw"})
    _tiny_onnx(fresh / "model.onnx", rows=ROWS[:-1] + [(610, 30, 20, 20, 0.44, 0.1, 0.1)])  # other sha
    _raises(lambda: export.run(fresh, OP, out_dir=TMP / "x9"), export.ExportError, "no test evaluation")
    # waived: recorded; parity runs against another eval of this model
    _eval_run("tiny2_def3_devall__probe_xprod", fresh, "probe_xprod", OP)
    export.run(fresh, OP, out_dir=TMP / "x10", no_test_eval=True)
    raw = json.loads((TMP / "x10" / "detector.json").read_text(encoding="utf-8"))
    assert raw["validation"]["test_eval_run"] is None and raw["validation"]["test_level1"] is None
    assert any("no_test_eval" in o for o in raw["validation"]["overrides"])
    assert raw["validation"]["parity"]["reference_eval_run"] == "tiny2_def3_devall__probe_xprod"
    # a test eval at ANOTHER operating point does not count
    other_op = _op("op_other_t.json", threshold=0.4)
    _eval_run("tiny2_def3_devall__test", fresh, "test", other_op)
    m = json.loads((settings.eval_dir / "tiny2_def3_devall__test" / "metrics.json").read_text())
    m["threshold"] = 0.4
    (settings.eval_dir / "tiny2_def3_devall__test" / "metrics.json").write_text(json.dumps(m))
    _raises(lambda: export.run(fresh, OP, out_dir=TMP / "x11"), export.ExportError, "another operating point")


def test_cascade_run_is_not_the_detectors_test_eval():
    """A cascade test run (detector + verifier) records the detector's sha but
    no operating_point block: it is neither the test eval nor a parity
    reference; a detector test run must have recorded this op's sha."""
    fresh = _bundle("tiny3_def3_devall", "dev_all", family="tiny_def3")
    _tiny_onnx(fresh / "model.onnx", rows=ROWS[:-1] + [(610, 30, 20, 20, 0.43, 0.1, 0.1)])  # own sha
    export._dev_photo_ids = lambda n: list(PHOTO_IDS)

    def rewrite(run, fn):
        for f in ("metrics.json", "predictions.json"):
            path = settings.eval_dir / run / f
            doc = json.loads(path.read_text(encoding="utf-8"))
            fn(doc)
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    def as_cascade(doc):
        doc["source"] = {**doc["source"], "kind": "cascade", "verifier_run": "v1",
                         "stacker_sha256": "e" * 64}
        doc.pop("operating_point", None)
        if "level1_verdict" in doc:
            doc["level1_verdict"] = {"n_positive": 99}

    _eval_run("tiny3_def3_devall+v1__test", fresh, "test", OP)
    rewrite("tiny3_def3_devall+v1__test", as_cascade)
    _raises(lambda: export.run(fresh, OP, out_dir=TMP / "x15"), export.ExportError, "no test evaluation")
    export.run(fresh, OP, out_dir=TMP / "x16", no_test_eval=True)
    raw = json.loads((TMP / "x16" / "detector.json").read_text(encoding="utf-8"))
    assert raw["validation"]["test_eval_run"] is None and raw["validation"]["test_level1"] is None
    assert raw["validation"]["parity"]["mode"].startswith("floor-equivalence"), raw["validation"]["parity"]
    # a detector test run at the right threshold that did not record the op's sha does not count
    _eval_run("tiny3_def3_devall__test", fresh, "test", OP)
    rewrite("tiny3_def3_devall__test", lambda doc: doc.pop("operating_point", None))
    _raises(lambda: export.run(fresh, OP, out_dir=TMP / "x17"), export.ExportError,
            "another operating point", "not recorded")
    _eval_run("tiny3_def3_devall__test", fresh, "test", OP)  # with the op block: it counts
    export.run(fresh, OP, out_dir=TMP / "x18")
    raw = json.loads((TMP / "x18" / "detector.json").read_text(encoding="utf-8"))
    assert raw["validation"]["test_eval_run"] == "tiny3_def3_devall__test"
    assert raw["validation"]["test_level1"]["n_positive"] == 3


def test_the_matching_test_look_is_found_among_further_looks():
    """A further test look is written to <bundle_run>__test__look<k>: export takes
    the detector test eval that recorded THIS op's sha (the latest of several),
    whatever look number it has; looks at another op never count."""
    fresh = _bundle("tiny4_def3_devall", "dev_all", family="tiny_def3")
    _tiny_onnx(fresh / "model.onnx", rows=ROWS[:-1] + [(610, 30, 20, 20, 0.42, 0.1, 0.1)])  # own sha
    other_op = _op("op_look_other_t.json", threshold=0.4)
    first = _eval_run("tiny4_def3_devall__test", fresh, "test", other_op)  # first look: another op
    m = json.loads((first / "metrics.json").read_text())
    (first / "metrics.json").write_text(json.dumps({**m, "threshold": 0.4}))
    _raises(lambda: export.run(fresh, OP, out_dir=TMP / "x19"), export.ExportError,
            "tiny4_def3_devall__test", "another operating point")
    export.run(fresh, OP, out_dir=TMP / "x24", no_test_eval=True)  # waived: recorded, with the other look
    raw = json.loads((TMP / "x24" / "detector.json").read_text(encoding="utf-8"))
    assert raw["validation"]["test_eval_run"] is None
    assert any("other operating points: tiny4_def3_devall__test" in o for o in raw["validation"]["overrides"])
    looks =[_eval_run(f"tiny4_def3_devall__test__look{k}", fresh, "test", OP) for k in (2, 3)]
    for k, d in enumerate([first, *looks]):  # look3 written last
        os.utime(d / "metrics.json", ns=(10 ** 18 + k, 10 ** 18 + k))
    export.run(fresh, OP, out_dir=TMP / "x20")
    raw = json.loads((TMP / "x20" / "detector.json").read_text(encoding="utf-8"))
    assert raw["validation"]["test_eval_run"] == "tiny4_def3_devall__test__look3", raw["validation"]
    assert raw["validation"]["overrides"] == [] and raw["validation"]["parity"]["mode"] == "reference"
    # an explicitly named test eval must still be one at this op
    _raises(lambda: export.run(fresh, OP, eval_run="tiny4_def3_devall__test", out_dir=TMP / "x21"),
            export.ExportError, "another operating point")
    export.run(fresh, OP, eval_run="tiny4_def3_devall__test__look2", out_dir=TMP / "x22")
    raw = json.loads((TMP / "x22" / "detector.json").read_text(encoding="utf-8"))
    assert raw["validation"]["test_eval_run"] == "tiny4_def3_devall__test__look2"


def test_a_fold_bundle_replaced_after_choose_op_makes_the_op_stale():
    """export runs the test eval's op <-> model check (run_eval.op_model_problems):
    a fold bundle that no longer holds the model the op was frozen from is a
    stale operating point."""
    fold0 = settings.bundles_dir / "tiny_def3_fold0" / "model.onnx"
    original = fold0.read_bytes()
    try:
        _tiny_onnx(fold0, rows=FOLD0_ROWS[:-1] + [(610, 30, 20, 20, 0.47, 0.1, 0.1)])  # re-imported
        _raises(lambda: export.run(DEVALL, OP, out_dir=TMP / "x23", force=True), export.ExportError,
                "tiny_def3_fold0 no longer holds", "stale operating point")
    finally:
        fold0.write_bytes(original)
    export.run(DEVALL, OP, out_dir=TMP / "x23")


def test_fold_model_and_v1_op_only_with_allow_fold_model():
    fold0 = settings.bundles_dir / "tiny_def3_fold0"
    export._dev_photo_ids = lambda n: list(PHOTO_IDS)  # no eval of this model: live floor mode
    v1 = _op("op_v1b.json", family=None, sources=None, target_sensitivity=None)
    export.run(fold0, v1, out_dir=TMP / "x12", allow_fold_model=True, no_test_eval=True)
    raw = json.loads((TMP / "x12" / "detector.json").read_text(encoding="utf-8"))
    over = " | ".join(raw["validation"]["overrides"])
    assert "allow_fold_model" in over and "v1 operating point" in over and "no_test_eval" in over
    assert raw["source"]["fold"] == "fold0" and raw["operating_point"]["family"] is None
    par = raw["validation"]["parity"]
    assert par["mode"].startswith("floor-equivalence") and par["verdict_mismatches"] == 0
    assert par["photos"] == 3


def test_parity_mismatch_blocks_the_export():
    def claim_io(photos):  # the "eval" claims photo 2 is i.O.
        photos[1]["scores"] = [min(s, 0.1) for s in photos[1]["scores"]]

    _eval_run("tiny_def3_devall__probe_xprod", DEVALL, "probe_xprod", OP, tamper=claim_io)
    _raises(lambda: export.run(DEVALL, OP, eval_run="tiny_def3_devall__probe_xprod",
                               out_dir=TMP / "x13", no_test_eval=True),
            export.ExportError, "parity FAILED", "synth0002")
    _raises(lambda: export.run(DEVALL, OP, eval_run="not_this_model", out_dir=TMP / "x14"),
            export.ExportError, "not an evaluation of this model")


def test_v1_reference_rounding_tolerance():
    """v1 predictions rounded scores to 5 decimals: a flip within 5e-6 of t is a
    rounding boundary, not a mismatch; in v2 (full precision) it IS a mismatch."""
    t = 0.6  # the Brücke score, float32(0.6) = 0.6000000238...

    def just_below(photos):
        for p in photos:
            p["scores"] = [0.599998 if abs(s - 0.6) < 1e-6 else s for s in p["scores"]]

    model = DEVALL / "model.onnx"
    _eval_run("round_v1", DEVALL, "probe_xprod", None, fmt=1, tamper=just_below)
    r = export.check_parity(model, "ultralytics", t, NAMES, "round_v1", 3)
    assert r["verdict_mismatches"] == 0 and len(r["rounding_boundary"]) == 3, r
    _eval_run("round_v2", DEVALL, "probe_xprod", None, fmt=2, tamper=just_below)
    r = export.check_parity(model, "ultralytics", t, NAMES, "round_v2", 3)
    assert r["verdict_mismatches"] == 3 and "rounding_boundary" not in r, r


def test_boundary_photo_selection_alternates_sides():
    photos = [{"capture_id": c, "scores": [s]} for c, s in
              (("a", 0.9), ("b", 0.52), ("c", 0.49), ("d", 0.1), ("e", 0.55))]
    order = [p["capture_id"] for p in export._pick_boundary_photos(photos, 0.5, 5)]
    assert order == ["b", "c", "e", "d", "a"], order


def test_floor_equivalence_property():
    """merge_by_class at floor t vs 0.001: identical scores >= t, class set and
    verdict (the CONTRACTS §1 argument for conf_floor = threshold)."""
    rng = random.Random(7)
    for trial in range(60):
        n = rng.randint(0, 450)
        boxes, scores, labels = [], [], []
        for _ in range(n):
            x, y = rng.uniform(0, 2500), rng.uniform(0, 1850)
            w, h = rng.uniform(4, 120), rng.uniform(4, 120)
            boxes.append((x, y, x + w, y + h))
            scores.append(rng.uniform(0.001, 1.0))
            labels.append(rng.randrange(3))
        for t in (0.05, 0.3, 0.5, 0.9):
            full = merge_by_class(boxes, scores, labels, 0.5, max_per_class=100)
            idx = [i for i, s in enumerate(scores) if s >= t]
            floor = merge_by_class([boxes[i] for i in idx], [scores[i] for i in idx],
                                   [labels[i] for i in idx], 0.5, max_per_class=100)
            above = sorted(((s, c) for s, c in zip(full[1], full[2]) if s >= t), reverse=True)
            assert above == sorted(zip(floor[1], floor[2]), reverse=True), (trial, t)
            v_full = photo_verdict(*full, t, NAMES)
            v_floor = photo_verdict(*floor, t, NAMES)
            assert (v_full["verdict"], v_full["classes"]) == (v_floor["verdict"], v_floor["classes"])
            if photo_score(full[1]) >= t:
                assert v_full["photo_score"] == v_floor["photo_score"]
            else:  # an i.O photo reports 0 at floor t (sub-threshold boxes are never produced)
                assert v_floor["photo_score"] == 0.0


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]  # file order
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
