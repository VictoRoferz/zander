"""
DEIM / D-FINE-S runner (ml/train/deim_run.py) without a GPU, torch or DEIM's
Colab-only deps: the generated override (absolute __include__, schedule and
EMA derived from epochs + view size, stage 2 off, no-Mosaic policy), the COCO
id rewrite 0..n-1, config validation, dispatch routing, the run configs, the
resume state machine + Drive mirror, the tie-tolerant parity check, a bundle
v2 built from FAKE training outputs (verify_bundle + meta + LICENSE_DEIM), and
DeimAdapter decoding a DEIM-style graph with labels unchanged.

When a real clone of the pinned DEIM commit is available (ZANDER_DEIM_SRC, or
ml_data/run_dirs/_deim_src of this checkout), every override key and the
licence are also checked against it; otherwise that part prints SKIP.

Run: ml/.venv/bin/python -B tests/test_deim_override.py
"""
import atexit
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "ml"
# Settings are created at import and mkdir their data root: keep the test's
# footprint in a temp folder, never in the repo's ml_data.
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="zander_test_deim_"))
atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)
_REAL_DEIM = Path(os.environ.get("ZANDER_DEIM_SRC") or ROOT / "ml_data" / "run_dirs" / "_deim_src")
os.environ["ML_DATA_ROOT"] = str(_TMP_ROOT / "ml_data")
for var in ("ZANDER_ABORT_AFTER_EPOCH", "ZANDER_MIRROR_INTERVAL_S", "ZANDER_WORKERS", "ZANDER_CACHE"):
    os.environ.pop(var, None)
sys.path.insert(0, str(ML))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from config.settings import settings  # noqa: E402
from train import bundle as B  # noqa: E402
from train import deim_run as D  # noqa: E402
from train import dispatch  # noqa: E402
from train import ultralytics_run as U  # noqa: E402

CLASSES = ["Nicht ausreichend Lot", "Brücke", "Fahne"]
DS = "ds_v2_def3"
FP = "f" * 64
CONFIGS = {
    "fold0": ML / "configs" / "matrix" / "deim_s_def3_fold0.yaml",
    "fold1": ML / "configs" / "matrix" / "deim_s_def3_fold1.yaml",
    "fold2": ML / "configs" / "matrix" / "deim_s_def3_fold2.yaml",
    "dev_all": ML / "configs" / "final" / "deim_s_def3_devall.yaml",
}
FAKE_LICENSE = ("Copyright (C) somebody.\n\nDEIM is licensed under the Apache License.\n\n"
                "                                Apache License\n"
                "                           Version 2.0, January 2004\n")


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(dir=_TMP_ROOT))


def _raises(exc, fn, *needles):
    try:
        fn()
    except exc as e:
        for n in needles:
            assert n in str(e), f"{n!r} not in {type(e).__name__}: {e}"
        return e
    raise AssertionError(f"{fn} did not raise {exc.__name__}")


@contextlib.contextmanager
def patched(*triples):
    """patched((module, "attr", value), ...) — restored afterwards."""
    saved = [(m, a, getattr(m, a)) for m, a, _ in triples]
    try:
        for m, a, v in triples:
            setattr(m, a, v)
        yield
    finally:
        for m, a, v in saved:
            setattr(m, a, v)


# ------------------------------------------------------------- fixtures


def _coco(n_images: int, first_id: int = 1, boxes_every: int = 2) -> dict:
    """Tiler-style COCO: category_id = class index + 1 (listed out of order)."""
    images, anns = [], []
    for i in range(n_images):
        iid = first_id + i
        images.append({"id": iid, "file_name": f"t{iid:04d}.jpg", "width": 640, "height": 640,
                       "capture_id": f"c{iid // 4}", "board": "A"})
        if i % boxes_every == 0:
            for k in range(1 + i % 3):
                anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": 1 + (i + k) % 3,
                             "bbox": [10.0 + k, 20.0, 28.0, 30.0], "area": 840.0, "iscrowd": 0})
    cats = [{"id": 3, "name": "Fahne"}, {"id": 1, "name": "Nicht ausreichend Lot"}, {"id": 2, "name": "Brücke"}]
    return {"images": images, "annotations": anns, "categories": cats}


def _fake_tiles(n_train: int = 40, n_val: int = 12, n_devall: int = 60) -> Path:
    root = settings.tiles_dir / DS
    if (root / "provenance.json").is_file():
        return root
    (root / "images").mkdir(parents=True)
    (root / "coco").mkdir()
    views = {"fold0_train": (n_train, 1), "fold1_train": (n_train, 1), "fold2_train": (n_train, 1),
             "fold0_val": (n_val, 500), "fold1_val": (n_val, 500), "fold2_val": (n_val, 500),
             "dev_all_train": (n_devall, 1)}
    for view, (n, first) in views.items():
        doc = _coco(n, first)
        (root / "coco" / f"{view}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        for im in doc["images"]:
            (root / "images" / im["file_name"]).write_bytes(b"jpg")
    prov = {"dataset_id": DS, "schema": "def3", "classes": CLASSES, "dataset_fingerprint": FP,
            "snapshot_id": "v2_2026-08-17_yolo", "split_id": "split_v2", "tiler_version": "v2",
            "augmentation": {"d4": True}}
    (root / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False), encoding="utf-8")
    return root


def _write_cfg(**over) -> Path:
    cfg = {"name": "deim_s_def3_fold0", "framework": "deim", "model": "deim_hgnetv2_s_coco",
           "adapter": "deim", "dataset_id": DS, "fold": "fold0", "epochs": 80, "imgsz": 640,
           "batch": 16, "seed": 20260806, "final_weights": "last"}
    cfg.update(over)
    for k in [k for k, v in over.items() if v is None]:
        del cfg[k]
    path = _tmp() / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


def _fake_deim_tree() -> Path:
    """A miniature DEIM checkout: the S config's include chain + LICENSE."""
    d = _tmp() / "DEIM"
    (d / "configs" / "deim_dfine").mkdir(parents=True)
    (d / "configs" / "base").mkdir()
    (d / "configs" / "deim_dfine" / "deim_hgnetv2_s_coco.yml").write_text(
        "__include__: ['./dfine_hgnetv2_s_coco.yml', '../base/deim.yml']\n"
        "epoches: 132\nflat_epoch: 64\nlist_key: [1, 2]\n", encoding="utf-8")
    (d / "configs" / "deim_dfine" / "dfine_hgnetv2_s_coco.yml").write_text(
        "__include__: ['../base/dataloader.yml']\nepoches: 72\nnum_classes: 80\nlist_key: [9, 9, 9]\n"
        "train_dataloader:\n  collate_fn:\n    stop_epoch: 72\n    base_size_repeat: 20\n", encoding="utf-8")
    (d / "configs" / "base" / "dataloader.yml").write_text(
        "train_dataloader:\n  total_batch_size: 32\n  collate_fn:\n    stop_epoch: 72\n    base_size: 640\n",
        encoding="utf-8")
    (d / "configs" / "base" / "deim.yml").write_text(
        "train_dataloader:\n  collate_fn:\n    mixup_prob: 0.5\n    stop_epoch: 50\nwarmup_iter: 2000\n",
        encoding="utf-8")
    (d / "LICENSE").write_text(FAKE_LICENSE, encoding="utf-8")
    return d


def _fake_deim_onnx(path: Path, labels, boxes, scores, names=None, out_names=("labels", "boxes", "scores")):
    """A DEIM-deploy-shaped graph: inputs (images, orig_target_sizes), outputs
    (labels int64 [N,k], boxes float [N,k,4], scores float [N,k]) as constants."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    k = len(scores)
    arrays = [np.asarray([labels], dtype=np.int64), np.asarray([boxes], dtype=np.float32),
              np.asarray([scores], dtype=np.float32)]
    nodes = [helper.make_node("Constant", [], [n], value=numpy_helper.from_array(a, name=f"{n}_c"))
             for n, a in zip(out_names, arrays)]
    inputs = [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, 640, 640]),
              helper.make_tensor_value_info("orig_target_sizes", TensorProto.INT64, ["N", 2])]
    outputs = [helper.make_tensor_value_info(out_names[0], TensorProto.INT64, ["N", k]),
               helper.make_tensor_value_info(out_names[1], TensorProto.FLOAT, ["N", k, 4]),
               helper.make_tensor_value_info(out_names[2], TensorProto.FLOAT, ["N", k])]
    model = helper.make_model(helper.make_graph(nodes, "fake_deim", inputs, outputs),
                              opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    if names is not None:
        model.metadata_props.add(key="names", value=repr({i: n for i, n in enumerate(names)}))
    onnx.save(model, str(path))
    return path


LABELS = [0, 1, 2, 1, 0, 2]
BOXES = [[10, 10, 38, 40], [100, 100, 130, 120], [200, 50, 230, 90], [5, 5, 9, 9],
         [300, 300, 330, 330], [0, 0, 1, 1]]
SCORES = [0.91, 0.75, 0.55, 0.30, 0.10, 0.001]


# ------------------------------------------------------------ override


def test_include_is_absolute_and_resolves_outside_the_deim_tree():
    tiles = _fake_tiles()
    deim = _fake_deim_tree()
    eff = D.resolve_config(B.load_config(_write_cfg()))
    layout = D.prepare_dataset_layout(eff, _tmp() / "dataset", CLASSES, write=False)
    sched = D.schedule_for(eff["epochs"], layout.n_train_images, eff["batch"])
    ov = D.build_override(eff, sched, layout, deim_dir=deim, output_dir=_tmp() / "out")
    inc = ov["__include__"]
    assert isinstance(inc, list) and len(inc) == 1 and os.path.isabs(inc[0]), inc
    assert inc[0].endswith("configs/deim_dfine/deim_hgnetv2_s_coco.yml"), inc
    assert Path(inc[0]).is_file()
    assert str(tiles) in ov["train_dataloader"]["dataset"]["img_folder"]
    # the override lives OUTSIDE the DEIM tree (run_dir): the include still resolves
    elsewhere = _tmp() / "run_dir" / "deim_override.yml"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text(D.render_override(ov, "test"), encoding="utf-8")
    assert yaml.safe_load(elsewhere.read_text(encoding="utf-8")) == ov, "render_override must round-trip"
    resolved = D.load_deim_config(elsewhere)
    assert "__include__" not in resolved
    # DEIM's merge order: includes in order, then the file; nested dicts merge, lists replace
    assert resolved["epoches"] == sched.epochs and resolved["num_classes"] == 3
    assert resolved["warmup_iter"] == sched.warmup_iter
    assert resolved["list_key"] == [1, 2]
    cf = resolved["train_dataloader"]["collate_fn"]
    assert cf["base_size"] == 640 and cf["stop_epoch"] == sched.epochs and cf["mixup_prob"] == 0.0
    assert resolved["train_dataloader"]["total_batch_size"] == 16
    # why it must be absolute: a relative include is joined with the OVERRIDE's folder
    rel = elsewhere.parent / "relative.yml"
    rel.write_text("__include__: ['../configs/deim_dfine/deim_hgnetv2_s_coco.yml']\n", encoding="utf-8")
    _raises(FileNotFoundError, lambda: D.load_deim_config(rel))
    # the text never wraps a path with spaces and the header is ASCII (DEIM opens without encoding)
    spaced = D.render_override({"__include__": ["/a b/c d.yml"], "x": {"img_folder": "/p q/" + "r" * 200}})
    assert all(ord(ch) < 128 for ch in spaced.split("__include__")[0])
    assert yaml.safe_load(spaced)["x"]["img_folder"] == "/p q/" + "r" * 200


def test_schedule_consistent_E80_E120_and_scaled_to_the_view():
    cases = ((80, 3338, 16), (120, 3338, 16), (80, 4606, 16), (80, 3338, 32), (80, 3031, 16))
    for E, n, batch in cases:
        s = D.schedule_for(E, n, batch)
        no_aug = max(2, int(0.1 * E + 0.5))
        assert s.epochs == E and s.no_aug_epoch == no_aug, s
        assert s.stop_epoch == E - no_aug and s.flat_epoch == (E - no_aug) // 2, s
        assert s.policy == (4, s.flat_epoch, s.stop_epoch), s
        assert 4 <= s.flat_epoch <= s.stop_epoch < E, s
        assert s.collate_stop_epoch == E, "stage 2 (best_stg1 reloads) must never trigger"
        assert s.iters_per_epoch == n // batch
        assert s.warmup_iter == s.ema_warmups == 2 * s.iters_per_epoch, "warm-ups = 2 epochs of OUR view"
        memory_epochs = 1.0 / (1.0 - s.ema_decay) / s.iters_per_epoch
        assert abs(memory_epochs - 2.7) < 0.01, (s.ema_decay, memory_epochs)
        # share of the exported EMA that still comes from the FIRST half of the run
        # (the flat high-LR phase): negligible with the scaled decay, but large with
        # DEIM's COCO decay 0.9999 on a view this small
        half = (E // 2) * s.iters_per_epoch
        assert s.ema_decay ** half < 1e-6, s.ema_decay ** half
        assert 0.9999 ** half > 0.2, 0.9999 ** half
        assert abs(s.lr - 0.0004 * batch / 32) < 1e-12 and abs(s.backbone_lr - 0.0002 * batch / 32) < 1e-12
    s80, s120 = D.schedule_for(80, 3338, 16), D.schedule_for(120, 3338, 16)
    assert (s80.no_aug_epoch, s80.stop_epoch, s80.flat_epoch, s80.policy) == (8, 72, 36, (4, 36, 72))
    assert (s120.no_aug_epoch, s120.stop_epoch, s120.flat_epoch, s120.policy) == (12, 108, 54, (4, 54, 108))
    assert s80.iters_per_epoch == 208 and s80.warmup_iter == 416
    assert D.schedule_for(80, 4606, 16).warmup_iter == 2 * (4606 // 16)
    tiny = D.schedule_for(2, 33, 4)  # smoke: degenerate but valid
    assert tiny.policy == (0, 0, 0) and tiny.stop_epoch == 0 and tiny.collate_stop_epoch == 2
    _raises(B.ConfigError, lambda: D.schedule_for(80, 10, 16), "empty")
    # ... and the override carries exactly these numbers in DEIM's key names
    _fake_tiles()
    eff = D.resolve_config(B.load_config(_write_cfg(epochs=120)))
    layout = D.prepare_dataset_layout(eff, _tmp(), CLASSES, write=False)
    s = D.schedule_for(120, layout.n_train_images, 16)
    ov = D.build_override(eff, s, layout, deim_dir=_fake_deim_tree(), output_dir=_tmp())
    assert (ov["epoches"], ov["flat_epoch"], ov["no_aug_epoch"], ov["warmup_iter"]) == (120, 54, 12, s.warmup_iter)
    assert ov["ema"] == {"type": "ModelEMA", "decay": s.ema_decay, "warmups": s.ema_warmups, "start": 0}
    tr = ov["train_dataloader"]
    assert tr["dataset"]["transforms"]["policy"] == {"name": "stop_epoch", "epoch": [4, 54, 108],
                                                     "ops": ["ColorJitter"]}
    assert tr["collate_fn"]["stop_epoch"] == 120 and tr["collate_fn"]["mixup_prob"] == 0.0
    assert tr["collate_fn"]["mixup_epochs"] == [4, 54] and tr["collate_fn"]["base_size_repeat"] == 20
    assert tr["total_batch_size"] == 16 and ov["optimizer"]["lr"] == s.lr
    assert ov["checkpoint_freq"] > 120 and ov["lrsheduler"] == "flatcosine" and ov["lr_gamma"] == 0.5
    assert ov["HGNetv2"] == {"pretrained": False}, "never DEIM's backbone download (exit() on failure)"
    assert ov["num_classes"] == 3 and ov["remap_mscoco_category"] is False


def test_augmentation_mirrors_the_ultralytics_policy():
    _fake_tiles()
    eff = D.resolve_config(B.load_config(_write_cfg()))
    layout = D.prepare_dataset_layout(eff, _tmp(), CLASSES, write=False)
    ov = D.build_override(eff, D.schedule_for(80, layout.n_train_images, 16), layout,
                          deim_dir=_fake_deim_tree(), output_dir=_tmp())
    ops = ov["train_dataloader"]["dataset"]["transforms"]["ops"]
    types = [o["type"] for o in ops]
    for banned in ("Mosaic", "RandomZoomOut", "RandomIoUCrop", "RandomPhotometricDistort"):
        assert banned not in types, f"{banned} shrinks/rotates 28-px defects or permutes RGB channels"
    assert types[-4:] == ["Resize", "SanitizeBoundingBoxes", "ConvertPILImage", "ConvertBoxes"], types
    jitter = ops[types.index("ColorJitter")]
    ua = U.DEFAULT_AUGMENT
    assert (jitter["brightness"], jitter["saturation"], jitter["hue"]) == (ua["hsv_v"], ua["hsv_s"], ua["hsv_h"])
    assert "contrast" not in jitter
    assert ops[types.index("RandomHorizontalFlip")]["p"] == ua["fliplr"]
    assert ops[types.index("RandomVerticalFlip")]["p"] == ua["flipud"]
    assert set(D.EXTRA_TRANSFORMS) == {"RandomVerticalFlip", "ColorJitter"}
    assert ov["train_dataloader"]["dataset"]["transforms"]["mosaic_prob"] == 0.0
    # switched off: no jitter op and an empty policy list; no multiscale
    eff2 = D.resolve_config(B.load_config(_write_cfg(augment={
        "brightness": 0.0, "saturation": 0.0, "hue": 0.0, "vflip": 0.0, "multiscale": False})))
    ov2 = D.build_override(eff2, D.schedule_for(80, layout.n_train_images, 16), layout,
                           deim_dir=_fake_deim_tree(), output_dir=_tmp())
    t2 = ov2["train_dataloader"]["dataset"]["transforms"]
    assert [o["type"] for o in t2["ops"]][:1] == ["RandomHorizontalFlip"] and t2["policy"]["ops"] == []
    assert ov2["train_dataloader"]["collate_fn"]["base_size_repeat"] is None


def test_override_key_problems_are_reported():
    base = {"epoches": 1, "train_dataloader": {"collate_fn": {"stop_epoch": 1}, "dataset": {"transforms": {}}}}
    ov = {"__include__": ["/x.yml"], "epoches": 80, "epochs": 80,
          "train_dataloader": {"collate_fn": {"stop_epoch": 80, "stopepoch": 1},
                               "dataset": {"transforms": {"ops": [{"type": "Resize"}, {"type": "Mosaik"}]}}}}
    problems = D.override_key_problems(ov, base)
    assert "epochs" in problems and "train_dataloader.collate_fn.stopepoch" in problems, problems
    assert "train_dataloader.dataset.transforms.ops" in problems  # not in this tiny base
    assert any("Mosaik" in p for p in problems) and not any("'Resize'" in p for p in problems)
    assert D.override_key_problems({"epoches": 3}, {"epoches": 1}) == []


def test_override_keys_and_licence_against_the_pinned_deim_clone():
    if D.source_state(_REAL_DEIM) != "ok":
        print(f"   SKIP: no clean clone of DEIM {D.DEIM_COMMIT[:7]} at {_REAL_DEIM} "
              "(set ZANDER_DEIM_SRC to a clone to run this check)")
        return
    _, sha = D.check_license(_REAL_DEIM)
    assert sha == D.DEIM_LICENSE_SHA256
    _fake_tiles()
    base = D.load_deim_config(_REAL_DEIM / D.BASE_CONFIG)
    for fold, path in list(CONFIGS.items()) + [("smoke", ML / "configs" / "deim_smoke_def3.yaml")]:
        eff = D.resolve_config(B.load_config(path))
        layout = D.prepare_dataset_layout(eff, _tmp(), CLASSES, write=False)
        ov = D.build_override(eff, D.schedule_for(eff["epochs"], max(layout.n_train_images, eff["batch"]),
                                                  eff["batch"]), layout, deim_dir=_REAL_DEIM, output_dir=_tmp())
        assert D.override_key_problems(ov, base) == [], (fold, D.override_key_problems(ov, base))
        assert Path(ov["__include__"][0]).is_file()
    # upstream values the policy relies on without overriding them
    assert base["train_dataloader"]["drop_last"] is True
    assert base["PostProcessor"]["num_top_queries"] == 300 and base["use_focal_loss"] is True
    assert base["train_dataloader"]["collate_fn"]["base_size_repeat"] == D.MULTISCALE_REPEAT
    assert base["optimizer"]["lr"] == D.BASE_LR and base["optimizer"]["params"][0]["lr"] == D.BASE_BACKBONE_LR
    ov_groups = [g["params"] for g in D.build_override(
        eff, D.schedule_for(2, 33, 4), layout, deim_dir=_REAL_DEIM, output_dir=_tmp())["optimizer"]["params"]]
    assert ov_groups == [g["params"] for g in base["optimizer"]["params"]], "same param-group regexes as DEIM-S"


# ------------------------------------------------------------- dataset


def test_coco_ids_rewritten_to_class_index():
    doc = _coco(9)
    out, mapping = D.remap_coco(doc, CLASSES)
    assert out["categories"] == [{"id": i, "name": n} for i, n in enumerate(CLASSES)]
    assert mapping == {1: 0, 2: 1, 3: 2}
    names = {c["id"]: c["name"] for c in doc["categories"]}
    assert len(out["annotations"]) == len(doc["annotations"]) and out["images"] == doc["images"]
    for old, new in zip(doc["annotations"], out["annotations"]):
        assert new["category_id"] == old["category_id"] - 1 == CLASSES.index(names[old["category_id"]])
        assert {k: v for k, v in new.items() if k != "category_id"} == \
               {k: v for k, v in old.items() if k != "category_id"}
    assert {a["category_id"] for a in out["annotations"]} <= {0, 1, 2}
    assert doc["categories"][0]["id"] == 3, "the input must not be mutated"
    # subsampling keeps only the chosen images' annotations
    keep = {1, 3}
    sub, _ = D.remap_coco(doc, CLASSES, keep_image_ids=keep)
    assert {im["id"] for im in sub["images"]} == keep and {a["image_id"] for a in sub["annotations"]} <= keep
    # refusals
    _raises(B.ConfigError, lambda: D.remap_coco(doc, ["Brücke", "Nicht ausreichend Lot", "Fahne"]), "!=")
    bad = json.loads(json.dumps(doc))
    bad["annotations"][0]["category_id"] = 7
    _raises(B.ConfigError, lambda: D.remap_coco(bad, CLASSES), "unknown category_id")
    bad = json.loads(json.dumps(doc))
    bad["annotations"][0]["image_id"] = 999
    _raises(B.ConfigError, lambda: D.remap_coco(bad, CLASSES), "unknown image")
    bad = json.loads(json.dumps(doc))
    bad["categories"].append({"id": 1, "name": "Fahne"})
    _raises(B.ConfigError, lambda: D.remap_coco(bad, CLASSES), "duplicate")


def test_dataset_layout_views_and_subsampling():
    tiles = _fake_tiles()
    out = _tmp() / "dataset"
    eff = D.resolve_config(B.load_config(_write_cfg()))
    lay = D.prepare_dataset_layout(eff, out, CLASSES)
    assert (lay.train_view, lay.val_view) == ("fold0_train", "fold0_val")
    assert lay.img_folder == (tiles / "images").resolve()
    tr = json.loads(lay.train_json.read_text(encoding="utf-8"))
    assert [c["id"] for c in tr["categories"]] == [0, 1, 2] and lay.n_train_images == len(tr["images"]) == 40
    assert lay.n_train_boxes == len(tr["annotations"]) and lay.category_map == {"1": 0, "2": 1, "3": 2}
    dev = D.resolve_config(B.load_config(_write_cfg(name="deim_s_def3_devall", fold="dev_all")))
    lay2 = D.prepare_dataset_layout(dev, _tmp(), CLASSES, write=False)
    assert (lay2.train_view, lay2.val_view, lay2.n_train_images) == ("dev_all_train", "fold0_val", 60)
    assert not lay2.train_json.exists(), "write=False writes nothing"
    small = D.resolve_config(B.load_config(_write_cfg(batch=4, extra={"fraction": 0.1, "val_fraction": 0.5})))
    a = D.prepare_dataset_layout(small, _tmp(), CLASSES, write=False)
    b = D.prepare_dataset_layout(small, _tmp(), CLASSES, write=False)
    assert a.n_train_images == 4 and a.n_val_images == 6 and a == b or (a.n_train_images, a.n_val_images) == \
        (b.n_train_images, b.n_val_images)
    (tiles / "images" / "t0001.jpg").unlink()
    try:
        _raises(FileNotFoundError, lambda: D.prepare_dataset_layout(eff, _tmp(), CLASSES, write=False), "missing")
    finally:
        (tiles / "images" / "t0001.jpg").write_bytes(b"jpg")


# -------------------------------------------------------------- config


def test_resolve_config_validation_and_overrides():
    raw = B.load_config(_write_cfg())
    eff = D.resolve_config(raw)
    assert eff["augment"] == D.DEFAULT_AUGMENT and eff["extra"] == {} and eff["final_weights"] == "last"
    eff = D.resolve_config(raw, {"augment.hflip": 0.25, "epochs": 60})
    assert eff["augment"]["hflip"] == 0.25 and eff["augment"]["vflip"] == 0.5 and eff["epochs"] == 60
    eff = D.resolve_config(raw, {"augment": {"vflip": 0.0}})  # nested dicts deep-merge (cli.py --set)
    assert eff["augment"] == dict(D.DEFAULT_AUGMENT, vflip=0.0)
    assert B.config_sha256(D.resolve_config(raw)) == B.config_sha256(D.resolve_config(dict(raw)))
    for over, needle in (
        ({"final_weights": "best"}, "selection leak"),
        ({"imgsz": 320}, "imgsz"),
        ({"fold": "fold1"}, "says fold0"),
        ({"framework": "ultralytics"}, "framework"),
        ({"model": "deim_hgnetv2_x"}, "model must be"),
        ({"extra": {"patience": 0}}, "unknown extra"),
        ({"extra": {"fraction": 0}}, "extra.fraction"),
        ({"augment": {"mosaic": 1.0}}, "unknown augment"),
        ({"augment": {"hue": 0.9}}, "augment.hue"),
        ({"deterministic": True}, "unknown top-level"),
        ({"batch": 0}, "batch"),
    ):
        _raises(B.ConfigError, lambda: D.resolve_config(B.load_config(_write_cfg(**over))), needle)
    _raises(B.ConfigError, lambda: D.resolve_config(raw, {"model": "deim_hgnetv2_s_random"}), "identity")
    _raises(B.ConfigError, lambda: D.resolve_config(raw, {"extra.val": False}), "unknown extra")


def test_run_configs_parse_and_the_folds_are_identical():
    effs = {}
    for fold, path in CONFIGS.items():
        raw = B.load_config(path)
        parsed = B.parse_run_name(raw["name"])
        assert parsed is not None, raw["name"]
        assert (parsed.family, parsed.arch, parsed.schema, parsed.fold) == ("deim_s_def3", "deim_s", "def3", fold)
        assert B.family_of(raw["name"]) == "deim_s_def3" and B.arch_of(raw["name"]) == "deim_s"
        eff = D.resolve_config(raw)
        assert (eff["framework"], eff["adapter"], eff["dataset_id"], eff["fold"]) == ("deim", "deim", DS, fold)
        assert (eff["epochs"], eff["batch"], eff["imgsz"], eff["final_weights"]) == (80, 16, 640, "last")
        assert eff["model"] == "deim_hgnetv2_s_coco" and eff["seed"] == 20260806
        effs[fold] = eff
    strip = lambda e: {k: v for k, v in e.items() if k not in ("name", "fold")}  # noqa: E731
    assert strip(effs["fold0"]) == strip(effs["fold1"]) == strip(effs["fold2"]) == strip(effs["dev_all"])
    text = [CONFIGS[f].read_text(encoding="utf-8").splitlines() for f in ("fold0", "fold1", "fold2")]
    diff = [(i, a, b, c) for i, (a, b, c) in enumerate(zip(*text)) if not a == b == c]
    assert len(text[0]) == len(text[1]) == len(text[2]) and [d[1].split(":")[0] for d in diff] == ["name", "fold"], diff
    yolo = B.load_config(ML / "configs" / "matrix" / "yolo26n_def3_fold0.yaml")
    assert yolo["seed"] == effs["fold0"]["seed"] and yolo["dataset_id"] == DS
    import registry

    assert registry.family_from_run_name("deim_s_def3_devall") == "deim_s_def3"
    # train/matrix.py resolves DEIM configs to a config sha (skip checks) and groups ETAs by arch
    from train import matrix

    items, errors = matrix.load_items([str(CONFIGS[f]) for f in ("fold0", "fold1", "fold2")], {})
    assert not errors, errors
    for it, fold in zip(items, ("fold0", "fold1", "fold2")):
        assert it.config_sha == B.config_sha256(effs[fold]) and it.arch == "deim_s" and it.epochs == 80
    smoke = D.resolve_config(B.load_config(ML / "configs" / "deim_smoke_def3.yaml"))
    assert B.parse_run_name(smoke["name"]) is None, "a smoke run is never pooled with the fold runs"
    assert smoke["model"] == "deim_hgnetv2_s_random"


def test_dispatch_routes_deim_configs_to_run_from_config():
    path = _write_cfg()
    calls = []

    def fake_run(config_path, device=None, overrides=None, mirror_dir=None):
        calls.append((Path(config_path), device, overrides, mirror_dir))
        return _tmp() / "deim_s_def3_fold0_bundle.zip"

    def no_ultralytics(*a, **k):
        raise AssertionError("a deim config must never reach the ultralytics runner")

    with patched((D, "run_from_config", fake_run), (U, "run", no_ultralytics)):
        out = dispatch.run(path, device="0", overrides={"epochs": 5}, mirror_dir=Path("/m"))
        assert out.name == "deim_s_def3_fold0_bundle.zip"
        assert calls == [(path, "0", {"epochs": 5}, Path("/m"))], calls
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = dispatch.main(["--config", str(path), "--device", "cpu", "--set", "epochs=7"])
        assert rc == dispatch.EXIT_OK and buf.getvalue().strip().endswith("_bundle.zip"), buf.getvalue()
        assert calls[-1][1:3] == ("cpu", {"epochs": 7})
    assert dispatch.framework_of(B.load_config(path)) == "deim"
    assert dispatch.resolve(path) == D.resolve_config(B.load_config(path))
    best = _write_cfg(final_weights="best")
    rc = dispatch.main(["--config", str(best)])
    assert rc == dispatch.EXIT_CONFIG, "config errors are not retried by the matrix"
    # ... and reported as such even where DEIM's deps are missing (checked after the config)
    with patched((D, "missing_modules", lambda *a: ["faster_coco_eval"])):
        _raises(B.ConfigError, lambda: D.run_from_config(best), "selection leak")


def test_device_plan_and_commands():
    assert D.device_plan("0") == ([], {"CUDA_VISIBLE_DEVICES": "0"}, True)
    assert D.device_plan("cuda:1") == ([], {"CUDA_VISIBLE_DEVICES": "1"}, True)
    assert D.device_plan("cpu") == (["-d", "cpu"], {}, False)
    assert D.device_plan(None) == ([], {}, True)
    for bad in ("mps", "0,1", "tpu"):
        _raises(B.ConfigError, lambda: D.device_plan(bad), "cpu or ONE CUDA")
    deim, ov = Path("/deim"), Path("/run/deim_override.yml")
    cmd, env = D.train_command(deim, ov, seed=7, device="0", tuning=Path("/w.pth"))
    tail = cmd[cmd.index("--") + 1:]
    assert cmd[3:5] == ["--_worker", "train"] and tail == ["-c", str(ov), "--seed", "7", "--use-amp", "-t", "/w.pth"]
    assert env["CUDA_VISIBLE_DEVICES"] == "0" and env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    cmd, _ = D.train_command(deim, ov, seed=7, device="cpu", resume=Path("/r.pth"), tuning=Path("/w.pth"))
    tail = cmd[cmd.index("--") + 1:]
    assert tail == ["-c", str(ov), "--seed", "7", "-d", "cpu", "-r", "/r.pth"], "resume wins over -t, no AMP on cpu"
    exp = D.export_command(deim, ov, Path("/ck.pth"), Path("/out"))
    assert exp[3:5] == ["--_worker", "export"] and exp[-6:] == ["--config", str(ov), "--checkpoint", "/ck.pth",
                                                                 "--out", "/out"]


def test_licence_check():
    deim = _fake_deim_tree()
    sha = hashlib.sha256(FAKE_LICENSE.encode("utf-8")).hexdigest()
    with patched((D, "DEIM_LICENSE_SHA256", sha)):
        assert D.check_license(deim) == (deim / "LICENSE", sha)
    _raises(B.ConfigError, lambda: D.check_license(deim), "sha256")
    nc = _tmp()
    text = FAKE_LICENSE + "\nThis software may be used for non-commercial purposes only.\n"
    (nc / "LICENSE").write_text(text, encoding="utf-8")
    with patched((D, "DEIM_LICENSE_SHA256", hashlib.sha256(text.encode()).hexdigest())):
        _raises(B.ConfigError, lambda: D.check_license(nc), "non-commercial")
    (nc / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    _raises(B.ConfigError, lambda: D.check_license(nc), "not the Apache")
    _raises(B.ConfigError, lambda: D.check_license(_tmp()), "missing")


# --------------------------------------------------- resume + mirror


def _fake_ckpt(path: Path, last_epoch: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"last_epoch": last_epoch}), encoding="utf-8")
    return path


def _fake_probe(path, total):
    path = Path(path)
    if not path.is_file():
        return D.CkptState("missing")
    try:
        done = int(json.loads(path.read_text(encoding="utf-8"))["last_epoch"]) + 1
    except (ValueError, KeyError):
        return D.CkptState("corrupt", detail="bad")
    return D.CkptState("finished" if done >= total else "resumable", done)


def _log(path: Path, epochs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"epoch": e, "train_loss": 1.0 / (e + 1)}) + "\n" for e in epochs),
                    encoding="utf-8")


def test_plan_start_resume_rules():
    sha = "a" * 64
    side = lambda d, s=sha, fp=FP: U.write_sidecar(d, {"format": 1, "config_sha256": s, "attempts": [],  # noqa: E731
                                                      "dataset_fingerprint": fp})
    # local resumable (2) vs mirror resumable (3): the mirror is more advanced -> restored
    root = _tmp()
    run, mirror = root / "run", root / "mirror"
    _fake_ckpt(run / "output" / "last.pth", 1)
    _log(run / "output" / "log.txt", range(2))
    side(run)
    _fake_ckpt(mirror / "last.pt", 2)
    _log(mirror / "log.txt", range(4))  # one epoch ahead of its checkpoint
    side(mirror)
    plan = D.plan_start(run, sha, 10, mirror, probe=_fake_probe, stamp="T", fingerprint=FP)
    assert (plan.kind, plan.source, plan.epochs_done) == ("resume", "mirror", 3), plan
    assert plan.moved_aside == root / "run.stale-T" and (root / "run.stale-T" / "output" / "last.pth").is_file()
    assert D.epochs_in_log(run / "output" / "log.txt") == 3, "the log is truncated to the checkpoint"
    assert json.loads((run / "output" / "last.pth").read_text())["last_epoch"] == 2
    # finished beats resumable; a local tie wins
    plan = D.plan_start(run, sha, 3, mirror, probe=_fake_probe, fingerprint=FP)
    assert (plan.kind, plan.source) == ("finished", "local"), plan
    # config or dataset changed -> fresh, the old run_dir moved aside (never deleted)
    for kwargs in ({"config_sha": "b" * 64, "fingerprint": FP}, {"config_sha": sha, "fingerprint": "e" * 64}):
        root2 = _tmp()
        run2 = root2 / "run"
        _fake_ckpt(run2 / "output" / "last.pth", 5)
        side(run2)
        plan = D.plan_start(run2, kwargs["config_sha"], 10, None, probe=_fake_probe, stamp="S",
                            fingerprint=kwargs["fingerprint"])
        assert plan.kind == "fresh" and plan.moved_aside == root2 / "run.stale-S" and not run2.exists(), plan
        assert "changed" in plan.reason
    # corrupt / missing everywhere -> fresh; nothing at all -> fresh without moving anything
    plan = D.plan_start(_tmp() / "nothing", sha, 10, _tmp(), probe=_fake_probe)
    assert (plan.kind, plan.moved_aside) == ("fresh", None)
    assert D.truncate_log(_tmp() / "none.txt", 3) == 0


def test_supervisor_mirrors_every_epoch_and_the_abort_hook():
    run, mirror = _tmp() / "run", _tmp() / "mirror"
    ck = _fake_ckpt(run / "output" / "last.pth", 0)
    old = time.time() - 60
    os.utime(ck, (old, old))
    sidecar = {"format": 1, "config_sha256": "a" * 64, "attempts": []}
    attempt = {"n": 1, "start_epoch": 0, "end_epoch": 0, "status": "running"}
    sidecar["attempts"].append(attempt)
    os.environ["ZANDER_MIRROR_INTERVAL_S"] = "0"
    os.environ["ZANDER_ABORT_AFTER_EPOCH"] = "2"
    try:
        sup = D.Supervisor(run, mirror, sidecar, attempt, total_epochs=3, start_epoch=0)
        assert sup.observe() is None and not mirror.exists(), "no epoch finished yet"
        _log(run / "output" / "log.txt", [0])
        assert sup.observe() is None
        assert attempt["end_epoch"] == 1 and U.read_sidecar(run)["attempts"][0]["end_epoch"] == 1
        assert sorted(p.name for p in mirror.iterdir()) == ["last.pt", "log.txt", U.SIDECAR]
        # a checkpoint still being written is not copied; the next poll retries it
        _fake_ckpt(ck, 1)
        _log(run / "output" / "log.txt", [0, 1])
        assert sup.observe() == D.ABORT_EXIT_CODE, "epoch 2 of 3 reached the abort hook"
        assert json.loads((mirror / "last.pt").read_text())["last_epoch"] == 1, "abort mirrors without settling"
        # the final epoch is mirrored even with a long interval, and never aborts
        os.environ["ZANDER_MIRROR_INTERVAL_S"] = "3600"
        sup2 = D.Supervisor(run, mirror, sidecar, attempt, total_epochs=3, start_epoch=2)
        _fake_ckpt(ck, 2)
        _log(run / "output" / "log.txt", [0, 1, 2])
        assert sup2.observe() is None
        assert json.loads((mirror / "last.pt").read_text())["last_epoch"] == 2
        assert D.epochs_in_log(mirror / "log.txt") == 3
    finally:
        os.environ.pop("ZANDER_MIRROR_INTERVAL_S", None)
        os.environ.pop("ZANDER_ABORT_AFTER_EPOCH", None)
    fresh = _tmp() / "f.bin"
    fresh.write_bytes(b"x")
    assert D.copy_if_stable(fresh, _tmp() / "g.bin", settle_s=30.0) is False, "fresh file = maybe mid-write"
    assert D.copy_if_stable(fresh, _tmp() / "g.bin", settle_s=0.0) is True
    # torn / foreign log lines are ignored by the epoch clock
    log = _tmp() / "log.txt"
    log.write_text('{"epoch": 0}\n{"epoch": 1}\n{"epo', encoding="utf-8")
    assert D.epochs_in_log(log) == 2


# ------------------------------------------------------- process tree


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for(cond, timeout=10.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


_TREE = """
import subprocess, sys, time
g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
open(sys.argv[1], "w").write(str(g.pid))
if sys.argv[2] == "linger":
    time.sleep(120)
"""


def test_the_whole_process_tree_is_stopped_and_stragglers_swept():
    # DEIM's DataLoader workers outlive a killed trainer (blocked on a full
    # result pipe) and keep matrix.py's log pipe open -> the matrix hangs.
    if os.name != "posix":
        print("   SKIP: POSIX process groups only")
        return
    d = _tmp()
    script = d / "tree.py"
    script.write_text(_TREE, encoding="utf-8")
    pid_file = d / "g1"
    proc = D.spawn_tree([sys.executable, str(script), str(pid_file), "linger"], dict(os.environ), d,
                        stderr=__import__("subprocess").DEVNULL)  # the SIGINT tracebacks
    assert _wait_for(lambda: pid_file.is_file() and pid_file.read_text().strip())
    grandchild = int(pid_file.read_text())
    assert os.getpgid(proc.pid) == proc.pid != os.getpgrp(), "own process group"
    D.stop_tree(proc, grace_s=2.0)
    assert proc.poll() is not None
    assert _wait_for(lambda: not _alive(grandchild), 5.0), "the grandchild survived stop_tree"
    # a subprocess that exits normally but leaves a child behind: swept by run_subprocess
    pid_file2 = d / "g2"
    rc = D.run_subprocess([sys.executable, str(script), str(pid_file2), "exit"], dict(os.environ), d, poll_s=0.05)
    assert rc == 0
    straggler = int(pid_file2.read_text())
    assert _wait_for(lambda: not _alive(straggler), 5.0), "the straggler was not swept"


def test_a_worker_takes_its_group_down_when_the_runner_dies():
    if os.name != "posix":
        print("   SKIP: POSIX process groups only")
        return
    d = _tmp()
    (d / "worker.py").write_text(
        "import sys, time\n"
        f"sys.path.insert(0, {str(ML)!r})\n"
        "from train import deim_run as D\n"
        "assert D.watch_parent(poll_s=0.1) is not None\n"
        "open(sys.argv[1], 'w').write('started')\n"
        "time.sleep(60)\n", encoding="utf-8")
    (d / "runner.py").write_text(
        "import os, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]], start_new_session=True)\n"
        "open(sys.argv[3], 'w').write(str(child.pid))\n"
        "import time\n"
        "end = time.time() + 30\n"
        "while not os.path.exists(sys.argv[2]) and time.time() < end:\n"
        "    time.sleep(0.05)\n"
        "os._exit(0)  # the runner dies without any cleanup\n", encoding="utf-8")
    started, pid_file = d / "started", d / "pid"
    subprocess_run = __import__("subprocess").run
    out = subprocess_run([sys.executable, str(d / "runner.py"), str(d / "worker.py"), str(started), str(pid_file)],
                         env=dict(os.environ), timeout=60)
    assert out.returncode == 0 and started.is_file(), "the worker never started"
    worker = int(pid_file.read_text())
    assert _wait_for(lambda: not _alive(worker), 10.0), "an orphaned worker kept running"
    assert D.watch_parent() is None, "a process that does not lead its own session is never guarded"


# ------------------------------------------------------------- parity


def test_compare_detections_is_tie_tolerant():
    ref = (LABELS, BOXES, SCORES)
    ok = D.compare_detections(ref, ref)
    assert ok["ok"] and ok["mismatches"] == 0 and ok["compared"] == 5, ok  # >= floor, above the k-th score
    # queries with equal scores may come out of TopK in any order
    tied_s = [0.8, 0.8, 0.8, 0.2, 0.001]
    tied_l = [0, 1, 0, 2, 0]
    tied_b = [[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30], [1, 1, 2, 2], [0, 0, 1, 1]]
    perm = [2, 0, 1, 3, 4]
    r = D.compare_detections((tied_l, tied_b, tied_s),
                             ([tied_l[i] for i in perm], [tied_b[i] for i in perm], [tied_s[i] for i in perm]))
    assert r["ok"] and r["compared"] == 4, r
    moved = [list(b) for b in BOXES]
    moved[1] = [b + 12 for b in moved[1]]
    assert not D.compare_detections(ref, (LABELS, moved, SCORES))["ok"]
    relabel = list(LABELS)
    relabel[0] = 2
    assert D.compare_detections(ref, (relabel, BOXES, SCORES))["mismatches"] == 1
    shifted = [s + 0.01 for s in SCORES]
    assert not D.compare_detections(ref, (LABELS, BOXES, shifted))["ok"]
    tiny = [s + 1e-6 for s in SCORES]
    r = D.compare_detections(ref, (LABELS, [[v + 0.01 for v in b] for b in BOXES], tiny))
    assert r["ok"] and r["max_box_diff_px"] <= 0.0101 and r["max_score_diff"] < 1e-5, r
    # a near-tie at the top-k cutoff may drop a box from one side only: not compared
    flat = [0.02 + 1e-5 * i for i in range(6)][::-1]
    r = D.compare_detections((LABELS, BOXES, flat), (LABELS, moved, flat))
    assert r["ok"] and r["compared"] == 0, r


def test_parity_matches_degenerate_and_tiny_boxes_by_coordinates():
    # D-FINE decodes inverted and zero-area boxes. Torch and ONNX agree on them
    # to ~0.02 px, but their IoU is 0 even against their own twin: a 1-epoch
    # fine-tune from the COCO init failed the gate with 12 such "mismatches".
    labels = [0, 1, 2, 0, 1, 2]
    boxes = [[609.8, 556.5, 639.2, 465.4],    # y inverted (seen in that run)
             [100.0, 100.0, 100.0, 130.0],    # zero width
             [50.0, 60.0, 40.0, 50.0],        # both inverted: positive "area", IoU 0
             [10.0, 10.0, 38.0, 40.0],        # proper
             [300.0, 300.0, 302.0, 302.0],    # proper but 2 px: a 0.05 px shift is IoU 0.91
             [0.0, 0.0, 1.0, 1.0]]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.001]
    near = [[v + 0.05 for v in b] for b in boxes]
    r = D.compare_detections((labels, boxes, scores), (labels, near, [s + 1e-6 for s in scores]))
    assert r["ok"] and (r["compared"], r["mismatches"], r["degenerate"]) == (5, 0, 3), r
    assert abs(r["max_box_diff_px"] - 0.05) < 1e-6 and 0.9 < r["min_iou"] < 0.92, r
    # still strict: a box that is genuinely elsewhere (1 px), or another label, is a mismatch
    for k in (0, 1, 4):
        far = [list(b) for b in boxes]
        far[k] = [v + 1.0 for v in far[k]]
        r = D.compare_detections((labels, boxes, scores), (labels, far, scores))
        assert not r["ok"] and r["mismatches"] == 1, (k, r)
    relabel = list(labels)
    relabel[0] = 1
    assert D.compare_detections((labels, boxes, scores), (relabel, boxes, scores))["mismatches"] == 1
    # only degenerate pairs: min_iou has nothing to report
    r = D.compare_detections((labels[:2] + [0], boxes[:2] + [boxes[5]], [0.9, 0.8, 0.001]),
                             (labels[:2] + [0], near[:2] + [near[5]], [0.9, 0.8, 0.001]))
    assert r["ok"] and r["degenerate"] == 2 and r["min_iou"] is None, r


def test_deim_adapter_decodes_a_deim_graph_with_labels_unchanged():
    from infer.onnx_adapters import DeimAdapter, load_adapter
    from infer.pipeline import TiledDetector

    path = _fake_deim_onnx(_tmp() / "model.onnx", LABELS, BOXES, SCORES, names=CLASSES)
    ad = load_adapter("deim", str(path))
    assert isinstance(ad, DeimAdapter) and (ad.image_input, ad.size_input) == ("images", "orig_target_sizes")
    tile = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes, scores, labels = ad.run(tile, conf_floor=0.3)
    keep = [i for i, s in enumerate(SCORES) if s >= 0.3]
    assert labels.dtype == np.int64 and labels.tolist() == [LABELS[i] for i in keep], labels
    assert np.allclose(boxes, np.asarray([BOXES[i] for i in keep], dtype=np.float32))
    assert np.allclose(scores, np.asarray([SCORES[i] for i in keep], dtype=np.float32))
    det = TiledDetector(path, "deim", conf_floor=0.3).detect(tile)
    assert sorted(det.labels) == sorted(LABELS[i] for i in keep)
    info = D.deim_onnx_info(path, CLASSES)
    assert info["layout"] == "deim" and info["output_names"] == ["labels", "boxes", "scores"]
    assert info["opset"] == 16 and info["class_names"] == CLASSES and len(info["sha256"]) == 64
    assert info["input_shapes"] == [["N", 3, 640, 640], ["N", 2]], info["input_shapes"]
    _raises(RuntimeError, lambda: D.deim_onnx_info(path, list(reversed(CLASSES))), "class names")
    wrong = _fake_deim_onnx(_tmp() / "w.onnx", LABELS, BOXES, SCORES, out_names=("classes", "boxes", "scores"))
    _raises(RuntimeError, lambda: D.deim_onnx_info(wrong, CLASSES), "not a DEIM deploy export")
    stamped = _fake_deim_onnx(_tmp() / "s.onnx", LABELS, BOXES, SCORES)
    D.stamp_onnx_metadata(stamped, CLASSES, "deim_s_def3_fold0")
    assert D.deim_onnx_info(stamped, CLASSES)["class_names"] == CLASSES


_FAKE_EXPORT_ONNX = '''
import json
import torch

def main(args):
    # DEIM's main() runs a batch-32 forward + trace here: what autograd records
    w = torch.nn.Parameter(torch.ones(2))
    y = (w * torch.ones(2)).sum()
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    with open(args.resume.replace(".pth", ".onnx"), "w") as f:
        json.dump({"grad_enabled": torch.is_grad_enabled(), "records_graph": y.requires_grad,
                   "check": args.check, "simplify": args.simplify, "ckpt_keys": sorted(ckpt)}, f)
'''


def test_the_export_worker_runs_deims_export_without_autograd():
    # DEIM's export_onnx.main() runs a batch-32 640-px forward and traces at
    # batch 32: with autograd on that peaked at 32.5 GB (3.5 GB without) and a
    # Colab VM without High-RAM OOM-kills the export after the whole training.
    try:
        import torch
        import torchvision  # noqa: F401  (the worker registers two torchvision transforms)
    except ImportError:
        print("   SKIP: torch/torchvision not installed")
        return
    deim = _tmp() / "DEIM"
    (deim / "engine" / "core").mkdir(parents=True)
    (deim / "engine" / "__init__.py").write_text("", encoding="utf-8")
    (deim / "engine" / "core" / "__init__.py").write_text(
        "GLOBAL_CONFIG = {}\n\ndef register():\n    def deco(cls):\n        GLOBAL_CONFIG[cls.__name__] = cls\n"
        "        return cls\n    return deco\n", encoding="utf-8")
    (deim / "tools" / "deployment").mkdir(parents=True)
    (deim / "tools" / "deployment" / "export_onnx.py").write_text(_FAKE_EXPORT_ONNX, encoding="utf-8")
    ckpt, out = _tmp() / "last.pth", _tmp() / "export"
    torch.save({"last_epoch": 2, "date": "x", "model": {"w": torch.zeros(1)}, "optimizer": {},
                "ema": {"module": {"w": torch.ones(1)}, "updates": 3}}, str(ckpt))
    cmd = D.export_command(deim, _tmp() / "override.yml", ckpt, out)
    run = __import__("subprocess").run([str(c) for c in cmd], capture_output=True, text=True, timeout=300,
                                       env=D.subprocess_env())
    assert run.returncode == 0, run.stderr[-2000:]
    got = json.loads((out / "model.onnx").read_text(encoding="utf-8"))
    assert got["grad_enabled"] is False and got["records_graph"] is False, got
    assert (got["check"], got["simplify"]) == (True, False), got
    assert got["ckpt_keys"] == ["date", "ema", "last_epoch", "zander"], "exported from the EMA-only checkpoint"
    stripped = torch.load(str(out / "last.pth"), map_location="cpu", weights_only=False)
    assert "optimizer" not in stripped and stripped["zander"]["weights"] == "ema"


# ------------------------------------------------------- the bundle


def _fake_heavy(deim: Path, calls: list, epochs: int):
    """run_subprocess / run_parity / probe / versions replaced by fakes that
    write what DEIM would: output/log.txt + last.pth per epoch, then the export."""

    def fake_run_subprocess(cmd, env, cwd, sup=None, poll_s=None):
        cmd = [str(c) for c in cmd]
        kind = cmd[cmd.index("--_worker") + 1]
        calls.append((kind, cmd))
        run_dir = Path(cwd)
        if kind == "train":
            tail = cmd[cmd.index("--") + 1:]
            assert "-c" in tail and Path(tail[tail.index("-c") + 1]).is_file()
            start = 0
            if "-r" in tail:
                start = json.loads(Path(tail[tail.index("-r") + 1]).read_text())["last_epoch"] + 1
            log = run_dir / "output" / "log.txt"
            for e in range(start, epochs):
                _fake_ckpt(run_dir / "output" / "last.pth", e)
                with open(log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"epoch": e, "train_loss": 1.0, "test_coco_eval_bbox": [0.1] * 12}) + "\n")
                if sup is not None:
                    sup.observe()
            return 0
        assert kind == "export"
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        _fake_ckpt(out / "last.pth", epochs - 1)
        _fake_deim_onnx(out / "model.onnx", LABELS, BOXES, SCORES)
        return 0

    def fake_parity(deim_dir, override, checkpoint, onnx_path, images, cwd):
        assert len(images) == D.PARITY_TILES and Path(onnx_path).is_file()
        return {"tiles": len(images), "ok": True, "compared": 7, "mismatches": 0, "max_score_diff": 1e-6}

    sha = hashlib.sha256(FAKE_LICENSE.encode("utf-8")).hexdigest()
    return (
        (D, "missing_modules", lambda *a: []),
        (D, "ensure_deim_source", lambda dest=None: deim),
        (D, "check_override_keys", lambda *a: None),
        (D, "DEIM_LICENSE_SHA256", sha),
        (D, "run_subprocess", fake_run_subprocess),
        (D, "run_parity", fake_parity),
        (D, "probe_checkpoint", _fake_probe),
        (D, "_versions", lambda: {"python": "3", "torch": "x", "cuda": None, "deim_commit": D.DEIM_COMMIT}),
        (U, "_gpu_name", lambda device: "fake-gpu"),
        (U, "_pip_freeze", lambda: "pkg==1\n"),
        (U, "_git_meta", lambda: {"git_sha": "0" * 40, "git_dirty": False}),
    )


def test_bundle_from_fake_training_outputs():
    _fake_tiles()
    deim = _fake_deim_tree()
    weights = D.weights_path("deim_hgnetv2_s_coco")
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"coco-init")
    cfg = _write_cfg()
    mirror_root = _tmp() / "mirror"
    calls: list = []
    with patched(*_fake_heavy(deim, calls, epochs=3)):
        zip_path = D.run_from_config(cfg, device="0", overrides={"epochs": 3}, mirror_dir=mirror_root)
        check = B.verify_bundle(zip_path)
        assert check.ok, check.reason
        meta = check.meta
        assert (meta["framework"], meta["adapter"], meta["family"], meta["fold"]) == ("deim", "deim", "deim_s_def3", "fold0")
        assert meta["onnx"]["layout"] == "deim" and meta["onnx_raw"] is None and meta["weights"] == "last"
        assert meta["versions"]["deim_commit"] == D.DEIM_COMMIT and meta["bundle_version"] == 2
        assert meta["epochs"] == 3 and meta["epochs_completed"] == 3 and meta["overrides"] == {"epochs": 3}
        assert meta["deim"]["commit"] == D.DEIM_COMMIT and meta["deim"]["license"] == "Apache-2.0"
        assert meta["init_weights"]["sha256"] == hashlib.sha256(b"coco-init").hexdigest()
        assert meta["parity"]["ok"] and meta["schedule"]["collate_stop_epoch"] == 3
        assert meta["dataset_view"]["category_map"] == {"1": 0, "2": 1, "3": 2}
        assert [a["status"] for a in meta["attempts"]] == ["finished"] and meta["gpu"] == "fake-gpu"
        for key in B.META_V1_KEYS + B.META_V2_KEYS:
            assert key in meta, key
        eff = D.resolve_config(B.load_config(cfg), {"epochs": 3})
        assert meta["config_sha256"] == B.config_sha256(eff)
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            assert {"config.yaml", "meta.json", "metrics.json", "model.onnx", "last.pt", "LICENSE_DEIM",
                    "args.yaml", "pip-freeze.txt", "deim_override.yml"} <= names, names
            assert zf.read("LICENSE_DEIM").decode("utf-8") == FAKE_LICENSE
            assert zf.read("config.yaml").decode("utf-8") == B.dump_config(eff)
            assert len(json.loads(zf.read("metrics.json"))["epochs"]) == 3
            args = yaml.safe_load(zf.read("args.yaml"))
            assert args["epoches"] == 3 and args["train_dataloader"]["collate_fn"]["stop_epoch"] == 3
            ov = yaml.safe_load(zf.read("deim_override.yml"))
            assert os.path.isabs(ov["__include__"][0])
        run_dir = settings.run_dirs_dir / "deim_s_def3_fold0"
        train_json = json.loads((run_dir / "dataset" / "train.json").read_text(encoding="utf-8"))
        assert [c["id"] for c in train_json["categories"]] == [0, 1, 2]
        tail = calls[0][1][calls[0][1].index("--") + 1:]
        assert "-t" in tail and tail[tail.index("-t") + 1] == str(weights) and "--use-amp" in tail
        assert [k for k, _ in calls] == ["train", "export"]
        mirror = mirror_root / "deim_s_def3_fold0"
        assert sorted(p.name for p in mirror.iterdir()) == ["last.pt", "log.txt", U.SIDECAR]
        assert U.read_sidecar(mirror)["attempts"][-1]["status"] == "finished"

        # rerun: the finished checkpoint is only exported + bundled again
        calls.clear()
        zip2 = D.run_from_config(cfg, device="0", overrides={"epochs": 3}, mirror_dir=mirror_root)
        assert [k for k, _ in calls] == ["export"] and B.verify_bundle(zip2).ok

        # a new VM: no local run_dir, a resumable mirror -> restored and resumed with -r
        shutil.rmtree(run_dir)
        for p in (zip_path, B.sha_path_for(zip_path)):
            p.unlink()
        _fake_ckpt(mirror / "last.pt", 1)
        _log(mirror / "log.txt", range(2))
        calls.clear()
        zip3 = D.run_from_config(cfg, device="0", overrides={"epochs": 3}, mirror_dir=mirror_root)
        tail = calls[0][1][calls[0][1].index("--") + 1:]
        assert "-r" in tail and "-t" not in tail, tail
        meta3 = B.read_bundle_meta(zip3)
        assert [a["resumed"] for a in meta3["attempts"]] == [False, True], meta3["attempts"]
        assert meta3["attempts"][-1]["start_epoch"] == 2 and meta3["epochs_completed"] == 3

        # training that exits 0 without finishing (e.g. DEIM's exit() paths) is a failure
        shutil.rmtree(run_dir)
        shutil.rmtree(mirror)

        def half_train(cmd, env, cwd, sup=None, poll_s=None):
            _fake_ckpt(Path(cwd) / "output" / "last.pth", 0)
            return 0

        with patched((D, "run_subprocess", half_train)):
            _raises(RuntimeError, lambda: D.run_from_config(cfg, device="0", overrides={"epochs": 3}),
                    "exited 0", "resumable")
    # missing init weights / deps are configuration errors (dispatch exit 2, never retried)
    weights.unlink()
    with patched(*_fake_heavy(deim, [], epochs=3)):
        _raises(FileNotFoundError, lambda: D.run_from_config(cfg), "--fetch-weights")
    with patched((D, "missing_modules", lambda *a: ["faster_coco_eval"])):
        _raises(B.ConfigError, lambda: D.run_from_config(cfg), "--install-deps")


def test_dry_run_prints_the_plan_without_side_effects():
    _fake_tiles()
    before = sorted(p.name for p in settings.run_dirs_dir.iterdir()) if settings.run_dirs_dir.exists() else []
    cfg = _write_cfg()
    text = D.dry_run(cfg, device="0")
    for needle in (D.DEIM_COMMIT, D.DEIM_REPO, str(D.deim_src_dir()), "__include__",
                   "configs/deim_dfine/deim_hgnetv2_s_coco.yml", "policy [4, 36, 72]", "stage 2 off",
                   "--_worker export", "--_worker train", " -t ", " -r ", "remap", "fold0_val",
                   "{'1': 0, '2': 1, '3': 2}", "EMA decay", "parity", "last.pt"):
        assert needle in text.replace("COCO category_id -> label", "remap"), needle
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert D.main(["--config", str(cfg), "--dry-run", "--device", "0"]) == 0
    assert D.DEIM_COMMIT in buf.getvalue()
    after = sorted(p.name for p in settings.run_dirs_dir.iterdir()) if settings.run_dirs_dir.exists() else []
    assert after == before, f"a dry run must not write anything ({before} -> {after})"


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
            import traceback

            traceback.print_exc()
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
