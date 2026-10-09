"""
Training config + bundle v2 plumbing (ml/train/bundle.py, ml/train/ultralytics_run.py,
ml/train/dispatch.py): effective config, family names, meta.json coverage of
docs/CONTRACTS.md §8.1, the dev_all/best refusal, the resume state machine
(stale run_dir moved aside, mirror restore) — CPU only, no training.

Run: ml/.venv/bin/python -B tests/test_train_config.py
"""
import atexit
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "ml"
# Settings are created at import and mkdir their data root: keep the test's
# footprint in a temp folder, never in the repo's ml_data.
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="zander_test_train_config_"))
atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)
os.environ["ML_DATA_ROOT"] = str(_TMP_ROOT / "ml_data")
sys.path.insert(0, str(ML))

from train import bundle as B  # noqa: E402
from train import dispatch  # noqa: E402
from train import ultralytics_run as U  # noqa: E402

MATRIX_CFG = {
    "name": "yolo26n_def3_fold0", "model": "yolo26n.pt", "adapter": "ultralytics",
    "dataset_id": "ds_v2_def3", "fold": "fold0", "epochs": 120, "imgsz": 640,
    "batch": 16, "seed": 20260806,
}
# docs/CONTRACTS.md §8.1, spelled out here on purpose: if the module's key
# lists ever lose a key, this test still fails.
CONTRACT_META_KEYS = [
    # v1
    "run_name", "framework", "adapter", "fold", "dataset_id", "schema", "classes",
    "dataset_fingerprint", "snapshot_id", "split_id", "tiler_version",
    "augmentation_offline", "augmentation_online", "ultralytics_version",
    "torch_version", "gpu", "git_sha", "git_dirty",
    # v2
    "bundle_version", "family", "weights", "epochs", "epochs_completed",
    "train_seconds", "attempts", "config_sha256", "overrides", "onnx", "onnx_raw",
    "versions", "created",
]
PROVENANCE = {
    "dataset_id": "ds_v2_def3", "schema": "def3",
    "classes": ["Nicht ausreichend Lot", "Brücke", "Fahne"],
    "dataset_fingerprint": "f" * 64, "snapshot_id": "v2_2026-08-17_yolo",
    "split_id": "split_v2", "tiler_version": "v2", "augmentation": {"d4": True},
}


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


def _onnx_info(file="model.onnx", layout="e2e"):
    shape = [[1, 300, 6]] if layout == "e2e" else [[1, 7, 8400]]
    return {"file": file, "sha256": "a" * 64, "opset": 17, "layout": layout, "output_shapes": shape}


# ------------------------------------------------------------ overrides


def test_parse_overrides_types():
    o = B.parse_overrides(["epochs=60", "extra.val=false", "augment.scale=0.3",
                           "extra.cache=disk", "export.opset=17", "extra.note="])
    assert o == {"epochs": 60, "extra.val": False, "augment.scale": 0.3,
                 "extra.cache": "disk", "export.opset": 17, "extra.note": None}, o
    _raises(B.ConfigError, lambda: B.parse_overrides(["epochs"]), "KEY=VALUE")
    _raises(B.ConfigError, lambda: B.parse_overrides(["=3"]), "KEY=VALUE")


def test_effective_config_deep_merge_and_dotted_overrides():
    defaults = {"augment": {"scale": 0.2, "mosaic": 1.0}, "extra": {"patience": 0}, "epochs": 120}
    cfg = {"name": "x_def3_fold0", "augment": {"scale": 0.3}}
    eff = B.effective_config(cfg, {"extra.val": False, "extra.deep.k": 1, "epochs": 60},
                             defaults=defaults)
    assert eff["augment"] == {"scale": 0.3, "mosaic": 1.0}  # config wins, defaults kept
    assert eff["extra"] == {"patience": 0, "val": False, "deep": {"k": 1}}
    assert eff["epochs"] == 60
    assert defaults["extra"] == {"patience": 0}, "defaults must not be mutated"
    # a plain key with a mapping value is deep-merged too
    eff2 = B.effective_config(cfg, {"augment": {"mosaic": 0.5}}, defaults=defaults)
    assert eff2["augment"] == {"scale": 0.3, "mosaic": 0.5}


def test_dump_is_canonical_and_sha_tracks_values():
    a = B.effective_config({"name": "n1", "epochs": 1, "batch": 8, "extra": {"b": 1, "a": 2}})
    b = B.effective_config({"extra": {"a": 2, "b": 1}, "batch": 8, "epochs": 1, "name": "n1"})
    assert B.dump_config(a) == B.dump_config(b)  # key order never matters
    assert B.config_sha256(a) == B.config_sha256(b)
    c = B.effective_config(a, {"epochs": 2})
    assert B.config_sha256(c) != B.config_sha256(a)
    text = B.dump_config(B.effective_config({"name": "n", "extra": {"cls": "Brücke"}}))
    assert "Brücke" in text  # unicode kept, not escaped


def test_nested_dict_overrides_deep_merge():
    # `cli.py train --set a.b=1` hands dispatch.run NESTED dicts: they must merge,
    # never wipe sibling keys (extra.patience, the augment policy, export.opset).
    eff = U.resolve_config(dict(MATRIX_CFG), {"extra": {"workers": 2}})
    assert eff["extra"] == {"patience": 0, "workers": 2}, eff["extra"]
    eff = U.resolve_config(dict(MATRIX_CFG), {"augment": {"scale": 0.3}, "export": {"raw_variant": "off"},
                                              "epochs": 60})
    assert eff["augment"] == dict(U.DEFAULT_AUGMENT, scale=0.3)
    assert eff["export"] == {"opset": 17, "raw_variant": "off"} and eff["epochs"] == 60
    # nested and dotted spellings are the same override, hence the same config sha
    a = U.resolve_config(dict(MATRIX_CFG), {"extra": {"val": False}})
    b = U.resolve_config(dict(MATRIX_CFG), {"extra.val": False})
    assert B.config_sha256(a) == B.config_sha256(b)
    # a config that sets its own extra keeps them under an override
    cfg = dict(MATRIX_CFG, extra={"fraction": 0.05, "plots": False})
    eff = U.resolve_config(cfg, {"extra": {"workers": 2}})
    assert eff["extra"] == {"patience": 0, "fraction": 0.05, "plots": False, "workers": 2}


def test_unknown_top_level_keys_rejected():
    _raises(B.ConfigError, lambda: B.effective_config({"name": "n", "epoch": 60}), "epoch")
    _raises(B.ConfigError, lambda: B.effective_config({"name": "n"}, {"epoch": 60}), "epoch")
    _raises(B.ConfigError, lambda: B.effective_config({"name": "n"}, {"bogus.x": 1}), "bogus")
    # a runner may widen the schema explicitly
    eff = B.effective_config({"name": "n", "deim": {"x": 1}},
                             allowed_keys=B.DEFAULT_ALLOWED_KEYS | {"deim"})
    assert eff["deim"] == {"x": 1}


def test_identity_keys_cannot_be_overridden():
    for key in ("name", "model", "fold", "dataset_id", "framework", "adapter"):
        _raises(B.ConfigError, lambda k=key: B.effective_config({"name": "n"}, {k: "x"}), "identity")


def test_name_must_be_file_safe():
    _raises(B.ConfigError, lambda: B.effective_config({"name": "a/b"}), "name")
    _raises(B.ConfigError, lambda: B.effective_config({}), "name")


# --------------------------------------------------------------- runner config


def test_resolve_config_applies_the_run_policy():
    eff = U.resolve_config(dict(MATRIX_CFG))
    assert eff["final_weights"] == "last"
    assert eff["extra"]["patience"] == 0
    assert eff["deterministic"] is True
    assert eff["export"] == {"opset": 17, "raw_variant": "auto"}
    assert eff["augment"] == U.DEFAULT_AUGMENT
    assert eff["framework"] == eff["adapter"] == "ultralytics"
    # configurable, but only through the config/overrides
    eff = U.resolve_config(dict(MATRIX_CFG), {"deterministic": False, "extra.patience": 30})
    assert eff["deterministic"] is False and eff["extra"]["patience"] == 30


def test_every_committed_config_resolves():
    configs = sorted((ML / "configs").rglob("*.yaml"))
    assert len(configs) >= 20, configs
    for path in configs:
        if path.name.startswith("deim_"):
            continue  # WP-deim's configs go through its own runner
        eff = dispatch.resolve(path)
        assert eff["final_weights"] == "last", path
        parsed = B.parse_run_name(eff["name"])
        if parsed is not None:
            assert parsed.fold == eff["fold"], path
            assert eff["dataset_id"].endswith("_" + parsed.schema), path
    finals = sorted(p for p in (ML / "configs" / "final").glob("*_devall.yaml")
                    if not p.name.startswith("deim_"))  # DEIM: tests/test_deim_override.py
    assert [p.stem for p in finals] == [
        "yolo11s_def3_devall", "yolo26n_def3_devall", "yolo26n_nal1_devall", "yolo26s_def3_devall"]
    for p in finals:
        eff = dispatch.resolve(p)
        assert eff["fold"] == "dev_all" and eff["epochs"] == 120 and eff["extra"]["val"] is False
    smoke = dispatch.resolve(ML / "configs" / "smoke_resume_def3.yaml")
    assert smoke["epochs"] == 2 and smoke["extra"]["fraction"] == 0.02 and smoke["extra"]["val"] is False


def test_matrix_and_final_names_follow_the_contract():
    # eval run names are <family>_fold<k> (CONTRACTS §8.2): every matrix/final
    # config name must parse, and agree with its fold and dataset schema.
    for sub in ("matrix", "final"):
        for path in sorted((ML / "configs" / sub).glob("*.yaml")):
            eff = dispatch.resolve(path)
            parsed = B.parse_run_name(eff["name"])
            assert parsed is not None, f"{path.name}: name {eff['name']!r} breaks the naming contract"
            assert path.stem == eff["name"], f"{path.name}: file name != run name"
            assert parsed.fold == eff["fold"] and eff["dataset_id"] == f"ds_v2_{parsed.schema}", path
            if eff["framework"] == "ultralytics":  # DEIM names its init in train/deim_run.MODELS
                assert eff["model"] == f"{parsed.arch}.pt", path


# choose-op / export compare these across the models of one family (WP-eval).
EQUIVALENCE_FIELDS = ("model", "dataset_id", "epochs", "imgsz", "batch", "augment", "final_weights",
                      "seed", "deterministic", "export")


def test_fold_models_of_a_family_are_equivalent():
    fams: dict = {}
    for path in sorted((ML / "configs" / "matrix").glob("*.yaml")):
        eff = dispatch.resolve(path)
        if eff["framework"] != "ultralytics":  # DEIM's fold equivalence: tests/test_deim_override.py
            continue
        fams.setdefault(B.family_of(eff["name"]), []).append(eff)
    assert sorted(fams) == ["yolo11s_def3", "yolo26n_def3", "yolo26n_nal1", "yolo26s_def3"]
    for fam, effs in fams.items():
        assert sorted(e["fold"] for e in effs) == ["fold0", "fold1", "fold2"], fam
        # the whole effective config is identical except the run identity
        stripped = [{k: v for k, v in e.items() if k not in ("name", "fold")} for e in effs]
        assert all(s == stripped[0] for s in stripped), f"{fam}: fold configs differ"
        final = dispatch.resolve(ML / "configs" / "final" / f"{fam}_devall.yaml")
        for field in EQUIVALENCE_FIELDS:
            assert final[field] == effs[0][field], f"{fam}: devall differs from the folds on {field}"


def test_dev_all_refuses_best():
    cfg = dict(MATRIX_CFG, name="yolo26n_def3_devall", fold="dev_all", final_weights="best")
    _raises(B.ConfigError, lambda: U.resolve_config(cfg), "dev_all refuses final_weights: best")
    _raises(B.ConfigError, lambda: U.resolve_config(
        dict(MATRIX_CFG, name="yolo26n_def3_devall", fold="dev_all"), {"final_weights": "best"}),
        "dev_all refuses")
    # a fold run may still ask for best (with a warning) ...
    assert U.resolve_config(dict(MATRIX_CFG, final_weights="best"))["final_weights"] == "best"
    # ... but nothing else
    _raises(B.ConfigError, lambda: U.resolve_config(dict(MATRIX_CFG, final_weights="ema")), "final_weights")


def test_runner_rejects_inconsistent_configs():
    _raises(B.ConfigError, lambda: U.resolve_config(dict(MATRIX_CFG, fold="fold1")), "says fold0")
    _raises(B.ConfigError, lambda: U.resolve_config(dict(MATRIX_CFG, extra={"data": "x.yaml"})), "data")
    _raises(B.ConfigError, lambda: U.resolve_config(dict(MATRIX_CFG, extra={"epochs": 3})), "epochs")
    _raises(B.ConfigError, lambda: U.resolve_config(
        dict(MATRIX_CFG, augment={"mosaic": 0.5}, extra={"mosaic": 0.0})), "mosaic")
    _raises(B.ConfigError, lambda: U.resolve_config(dict(MATRIX_CFG), {"export.opset": 99}), "opset")
    _raises(B.ConfigError, lambda: U.resolve_config(dict(MATRIX_CFG), {"export.raw_variant": "yes"}),
            "raw_variant")
    _raises(B.ConfigError, lambda: U.resolve_config(dict(MATRIX_CFG), {"export.fp16": True}), "export key")
    _raises(B.ConfigError, lambda: U.resolve_config(dict(MATRIX_CFG, adapter="deim")), "adapter")
    _raises(B.ConfigError, lambda: U.resolve_config(dict(MATRIX_CFG, epochs="120")), "epochs")


def test_runtime_knobs_come_from_env_only():
    assert U.runtime_from_env({}) == {}
    assert U.runtime_from_env({"ZANDER_WORKERS": "8", "ZANDER_CACHE": "disk"}) == {"workers": 8, "cache": "disk"}
    assert U.runtime_from_env({"ZANDER_CACHE": "false"}) == {"cache": False}
    _raises(B.ConfigError, lambda: U.runtime_from_env({"ZANDER_WORKERS": "-1"}), "ZANDER_WORKERS")
    _raises(B.ConfigError, lambda: U.runtime_from_env({"ZANDER_CACHE": "gpu"}), "ZANDER_CACHE")


# ------------------------------------------------------------------ names


def test_family_and_arch_of():
    assert B.family_of("yolo26n_def3_fold0") == "yolo26n_def3"
    assert B.family_of("yolo26n_def3_devall") == "yolo26n_def3"
    assert B.family_of("yolo26n_nal1_fold2") == "yolo26n_nal1"
    assert B.family_of("deim_s_def3_fold1") == "deim_s_def3"
    assert B.arch_of("deim_s_def3_fold1") == "deim_s"
    assert B.arch_of("yolo11s_def3_fold0") == "yolo11s"
    # free-form names are their own family: never pooled with the matrix
    assert B.family_of("yolo11n_smoke_def3") == "yolo11n_smoke_def3"
    assert B.family_of("yolo11s_def3_fold0_baseline") == "yolo11s_def3_fold0_baseline"
    p = B.parse_run_name("yolo26s_def3_devall")
    assert p == B.RunName("yolo26s_def3", "yolo26s", "def3", "dev_all"), p
    assert B.parse_run_name("yolo11n_smoke_def3") is None


# ------------------------------------------------------------------- meta


def _meta(**over):
    eff = U.resolve_config(dict(MATRIX_CFG))
    kw = dict(
        config=eff, provenance=PROVENANCE, framework="ultralytics", adapter="ultralytics",
        weights="last", epochs_completed=120,
        attempts=[{"gpu": "NVIDIA L4", "started": "2026-09-28T10:00:00Z", "seconds": 1800.0, "resumed": False},
                  {"gpu": "NVIDIA L4", "started": "2026-09-28T12:00:00Z", "seconds": 1500.5, "resumed": True}],
        onnx=_onnx_info(), onnx_raw=_onnx_info("model_raw.onnx", "raw"),
        versions={"python": "3.12.0", "torch": "2.8.0", "cuda": "12.6", "ultralytics": "8.4.115"},
        overrides={"extra.val": False}, git={"git_sha": "abc", "git_dirty": False},
    )
    kw.update(over)
    return eff, B.build_meta(**kw)


def test_meta_covers_contract_8_1():
    eff, meta = _meta()
    missing = [k for k in CONTRACT_META_KEYS if k not in meta]
    assert not missing, f"meta.json lacks contract keys {missing}"
    assert set(B.META_V1_KEYS) | set(B.META_V2_KEYS) <= set(meta)
    assert meta["bundle_version"] == 2
    assert meta["family"] == "yolo26n_def3" and meta["run_name"] == "yolo26n_def3_fold0"
    assert meta["weights"] == "last" and meta["epochs"] == 120 and meta["epochs_completed"] == 120
    assert meta["train_seconds"] == 3300.5
    assert meta["config_sha256"] == B.config_sha256(eff)
    assert meta["gpu"] == "NVIDIA L4" and meta["attempts"][1]["resumed"] is True
    for key in ("gpu", "started", "seconds", "resumed"):
        assert all(key in a for a in meta["attempts"]), key
    for info in (meta["onnx"], meta["onnx_raw"]):
        assert set(B.ONNX_INFO_KEYS) <= set(info)
    assert meta["onnx"]["layout"] == "e2e" and meta["onnx_raw"]["layout"] == "raw"
    assert {"python", "torch", "cuda", "ultralytics"} <= set(meta["versions"])
    assert meta["classes"] == PROVENANCE["classes"] and meta["split_id"] == "split_v2"
    assert meta["created"].endswith("Z")
    json.dumps(meta)  # serialisable as-is
    _, meta = _meta(onnx_raw=None)
    assert meta["onnx_raw"] is None


def test_meta_rejects_bad_inputs():
    _raises(ValueError, lambda: _meta(weights="ema"), "weights")
    _raises(ValueError, lambda: _meta(onnx=dict(_onnx_info(), layout="nms")), "layout")
    _raises(ValueError, lambda: _meta(onnx={"file": "model.onnx"}), "lacks")
    _raises(ValueError, lambda: _meta(extra={"family": "x"}), "collides")


# ----------------------------------------------------------------- bundle


def test_write_and_verify_bundle():
    d = _tmp()
    weights = d / "last.pt"
    weights.write_bytes(b"\x00" * 1000)
    _, meta = _meta()
    files = {"config.yaml": "name: x\n", "meta.json": json.dumps(meta), "metrics.json": "{}",
             "model.onnx": b"onnx", "last.pt": weights, "pip-freeze.txt": "a==1\n"}
    zip_path, sha = B.write_bundle(d / "yolo26n_def3_fold0_bundle.zip", files)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert sorted(names) == sorted(files) and all("/" not in n for n in names)  # flat
    marker = B.sha_path_for(zip_path)
    assert marker.name == "yolo26n_def3_fold0_bundle.zip.sha256"
    assert B.read_sha256_file(marker) == sha == B.sha256_file(zip_path)
    check = B.verify_bundle(zip_path)
    assert check.ok and check.sha256 == sha and check.meta["run_name"] == "yolo26n_def3_fold0"
    assert B.read_bundle_meta(zip_path)["bundle_version"] == 2
    assert not [p for p in d.iterdir() if p.name.startswith(".")], "temp files left behind"
    # tampering, a missing marker and a torn marker are all invalid
    marker.write_text("0" * 64 + "  x\n")
    assert not B.verify_bundle(zip_path).ok and "mismatch" in B.verify_bundle(zip_path).reason
    marker.write_text("garbage")
    assert not B.verify_bundle(zip_path).ok
    marker.unlink()
    assert "marker" in B.verify_bundle(zip_path).reason
    assert B.verify_bundle(d / "nope.zip").reason == "missing"


def test_verify_bundle_separates_io_errors_from_bad_bundles():
    # the matrix moves an INVALID bundle aside and retrains it; an unreadable
    # one (Drive mount dropped) must come back as io_error, never as invalid
    d = _tmp()
    zip_path, _ = B.write_bundle(d / "r_bundle.zip", {"config.yaml": "", "meta.json": "{}",
                                                       "metrics.json": "{}", "model.onnx": b""})
    marker = B.sha_path_for(zip_path)
    assert B.verify_bundle(zip_path).ok and not B.verify_bundle(zip_path).io_error
    real_sha = B.sha256_file
    B.sha256_file = lambda *a, **kw: (_ for _ in ()).throw(OSError(5, "Input/output error"))
    try:
        check = B.verify_bundle(zip_path)
    finally:
        B.sha256_file = real_sha
    assert not check.ok and check.io_error and "Input/output error" in check.reason, check
    if os.geteuid() != 0:  # root reads anything
        marker.chmod(0)
        try:
            check = B.verify_bundle(zip_path)
        finally:
            marker.chmod(0o644)
        assert not check.ok and check.io_error, check
    # real content problems stay "invalid" (io_error False)
    marker.write_text("0" * 64 + "  r_bundle.zip\n")
    assert not B.verify_bundle(zip_path).ok and not B.verify_bundle(zip_path).io_error
    marker.unlink()
    assert "marker" in B.verify_bundle(zip_path).reason and not B.verify_bundle(zip_path).io_error
    assert not B.verify_bundle(d / "nope.zip").io_error


def test_write_bundle_rejects_bad_members():
    d = _tmp()
    ok = {"config.yaml": "", "meta.json": "{}", "metrics.json": "{}", "model.onnx": b""}
    _raises(ValueError, lambda: B.write_bundle(d / "a.zip", {**ok, "sub/x.txt": "x"}), "flat")
    _raises(ValueError, lambda: B.write_bundle(d / "a.zip", {"config.yaml": ""}), "missing")
    assert not (d / "a.zip").exists()


# -------------------------------------------------------- resume state machine


def _fake_ckpt(path: Path, kind: str, epochs_done=None, epochs=10, stripped=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": kind, "epochs_done": epochs_done, "epochs": epochs,
                                "stripped": stripped}))


def fake_probe(pt: Path) -> "U.CkptState":
    if not Path(pt).is_file():
        return U.CkptState("missing")
    try:
        d = json.loads(Path(pt).read_text())
    except ValueError:
        return U.CkptState("corrupt", detail="unreadable")
    return U.CkptState(d["kind"], d["epochs_done"], d["epochs"], d["stripped"])


def _run_dir_with(d: Path, name: str, sha: str, kind=None, epochs_done=None, rows=0, stripped=False):
    run_dir = d / name
    run_dir.mkdir(parents=True)
    if kind:
        _fake_ckpt(run_dir / "weights" / "last.pt", kind, epochs_done, stripped=stripped)
    if sha:
        U.write_sidecar(run_dir, {"format": 1, "run_name": name, "config_sha256": sha,
                                  "attempts": [{"n": 1, "status": "running"}]})
    if rows:
        (run_dir / "results.csv").write_text(
            "epoch,time,loss\n" + "".join(f"{i},{i * 10},{1 / i:.3f}\n" for i in range(1, rows + 1)))
    (run_dir / "args.yaml").write_text("epochs: 10\n")
    return run_dir


def test_stale_run_dir_is_moved_aside():
    d = _tmp()
    run_dir = _run_dir_with(d, "r_def3_fold0", sha=None, rows=3)  # crashed before any checkpoint
    plan = U.prepare_run_dir(run_dir, "s" * 64, probe=fake_probe, stamp="20260928T120000Z")
    assert plan.kind == "fresh" and plan.source == "none"
    assert not run_dir.exists()
    assert plan.moved_aside == d / "r_def3_fold0.stale-20260928T120000Z"
    assert (plan.moved_aside / "results.csv").is_file(), "moved, never deleted"
    # a second stale dir in the same second gets a suffix instead of clobbering
    run_dir = _run_dir_with(d, "r_def3_fold0", sha=None)
    plan2 = U.prepare_run_dir(run_dir, "s" * 64, probe=fake_probe, stamp="20260928T120000Z")
    assert plan2.moved_aside.name == "r_def3_fold0.stale-20260928T120000Z-1"
    # nothing at all -> fresh without moving anything
    plan3 = U.prepare_run_dir(d / "never_ran", "s" * 64, probe=fake_probe)
    assert plan3.kind == "fresh" and plan3.moved_aside is None


def test_changed_config_or_unknown_provenance_is_not_resumed():
    d = _tmp()
    run_dir = _run_dir_with(d, "r1", sha="a" * 64, kind="resumable", epochs_done=4)
    plan = U.prepare_run_dir(run_dir, "b" * 64, probe=fake_probe)
    assert plan.kind == "fresh" and "config changed" in plan.reason and plan.moved_aside
    run_dir = _run_dir_with(d, "r2", sha=None, kind="resumable", epochs_done=4)  # no sidecar
    plan = U.prepare_run_dir(run_dir, "b" * 64, probe=fake_probe)
    assert plan.kind == "fresh" and "provenance" in plan.reason
    run_dir = _run_dir_with(d, "r3", sha="b" * 64)
    (run_dir / "weights").mkdir()
    (run_dir / "weights" / "last.pt").write_text("torn write")
    plan = U.prepare_run_dir(run_dir, "b" * 64, probe=fake_probe)
    assert plan.kind == "fresh" and "corrupt" in plan.reason


def test_local_resume_truncates_results_ahead_of_checkpoint():
    d = _tmp()
    run_dir = _run_dir_with(d, "r", sha="c" * 64, kind="resumable", epochs_done=3, rows=4)
    plan = U.prepare_run_dir(run_dir, "c" * 64, probe=fake_probe)
    assert (plan.kind, plan.source, plan.epochs_done, plan.moved_aside) == ("resume", "local", 3, None)
    rows = (run_dir / "results.csv").read_text().splitlines()
    assert [r.split(",")[0] for r in rows] == ["epoch", "1", "2", "3"], rows


def test_restore_from_mirror_on_a_fresh_vm():
    d = _tmp()
    mirror = d / "mirror" / "r"
    src = _run_dir_with(d / "old_vm", "r", sha="c" * 64, kind="resumable", epochs_done=5, rows=5)
    U.mirror_run_dir(src, mirror)
    assert sorted(p.name for p in mirror.iterdir()) == ["args.yaml", "last.pt", "results.csv", "zander_run.json"]
    run_dir = d / "new_vm" / "r"  # the new VM has no run_dir at all
    plan = U.prepare_run_dir(run_dir, "c" * 64, mirror_run=mirror, probe=fake_probe)
    assert (plan.kind, plan.source, plan.epochs_done) == ("resume", "mirror", 5)
    assert (run_dir / "weights" / "last.pt").is_file() and (run_dir / "results.csv").is_file()
    assert U.read_sidecar(run_dir)["config_sha256"] == "c" * 64
    # the more advanced checkpoint wins; the local run_dir is moved aside, not merged
    run_dir2 = _run_dir_with(d / "vm2", "r", sha="c" * 64, kind="resumable", epochs_done=2, rows=2)
    plan = U.prepare_run_dir(run_dir2, "c" * 64, mirror_run=mirror, probe=fake_probe)
    assert (plan.source, plan.epochs_done) == ("mirror", 5) and plan.moved_aside is not None
    # a mirror from another config is ignored
    plan = U.prepare_run_dir(d / "vm3" / "r", "e" * 64, mirror_run=mirror, probe=fake_probe)
    assert plan.kind == "fresh" and "config changed" in plan.reason


def test_finished_runs_skip_to_bundling():
    d = _tmp()
    run_dir = _run_dir_with(d, "r", sha="c" * 64, kind="finished", epochs_done=None, stripped=True, rows=10)
    plan = U.prepare_run_dir(run_dir, "c" * 64, probe=fake_probe)
    assert (plan.kind, plan.stripped) == ("finished", True)
    assert len((run_dir / "results.csv").read_text().splitlines()) == 11, "never truncated"
    # loop finished but final_eval never stripped (process died right after the last epoch)
    run_dir = _run_dir_with(d, "r2", sha="c" * 64, kind="finished", epochs_done=10)
    plan = U.prepare_run_dir(run_dir, "c" * 64, probe=fake_probe)
    assert (plan.kind, plan.stripped) == ("finished", False)
    # finished beats a resumable mirror
    mirror = d / "m" / "r3"
    U.mirror_run_dir(_run_dir_with(d / "x", "r3", sha="c" * 64, kind="resumable", epochs_done=9), mirror)
    run_dir = _run_dir_with(d, "r3", sha="c" * 64, kind="finished", stripped=True)
    plan = U.prepare_run_dir(run_dir, "c" * 64, mirror_run=mirror, probe=fake_probe)
    assert (plan.kind, plan.source) == ("finished", "local")


def test_truncate_results_csv():
    d = _tmp()
    p = d / "results.csv"
    p.write_text("epoch,time\n1,1\n2,2\n3,3\n")
    assert U.truncate_results_csv(p, 2) == 1
    assert p.read_text() == "epoch,time\n1,1\n2,2\n"
    assert U.truncate_results_csv(p, 5) == 0
    assert U.truncate_results_csv(d / "missing.csv", 1) == 0


def test_metrics_dedupe_by_epoch():
    d = _tmp()
    (d / "results.csv").write_text("epoch,time,loss\n1,1,0.9\n2,2,0.8\n2,2.5,0.7\n3,3,0.6\n")
    m = U._metrics_from_results_csv(d)
    assert [r["epoch"] for r in m["epochs"]] == [1.0, 2.0, 3.0]
    assert m["epochs"][1]["loss"] == 0.7 and m["final"]["loss"] == 0.6
    assert U._epochs_completed(m, {"attempts": []}) == 3
    assert U._epochs_completed({"epochs": []}, {"attempts": [{"end_epoch": 7}]}) == 7


def test_epoch_hook_updates_attempt_and_mirror():
    d = _tmp()
    run_dir = _run_dir_with(d, "r", sha="c" * 64, kind="resumable", epochs_done=1, rows=1)
    sidecar = U.read_sidecar(run_dir)
    attempt = {"n": 2, "status": "running", "start_epoch": 1, "end_epoch": 1}
    sidecar["attempts"].append(attempt)
    mirror = d / "mirror" / "r"
    saved = {k: os.environ.pop(k, None) for k in (U.ENV_ABORT, U.ENV_MIRROR_INTERVAL)}
    try:
        hook = U.EpochHook(run_dir, mirror, sidecar, attempt)
        assert hook.interval == U.DEFAULT_MIRROR_INTERVAL_S and hook.abort_after is None

        class Trainer:  # what Ultralytics hands the callback
            epoch, epochs = 1, 10

        hook(Trainer())
        assert attempt["end_epoch"] == 2 and attempt["seconds"] >= 0
        assert U.read_sidecar(run_dir)["attempts"][-1]["end_epoch"] == 2
        assert (mirror / "last.pt").is_file() and U.read_sidecar(mirror)["attempts"][-1]["end_epoch"] == 2
        # within the interval nothing is copied again ...
        (mirror / "last.pt").unlink()
        Trainer.epoch = 2
        hook(Trainer())
        assert not (mirror / "last.pt").exists()
        # ... but the final epoch is always mirrored
        Trainer.epoch = 9
        hook(Trainer())
        assert (mirror / "last.pt").is_file()
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_mirror_failure_is_never_fatal():
    d = _tmp()
    run_dir = _run_dir_with(d, "r", sha="c" * 64, kind="resumable", epochs_done=1)
    sidecar = U.read_sidecar(run_dir)
    attempt = {"n": 2}
    sidecar["attempts"].append(attempt)
    blocker = d / "not_a_dir"
    blocker.write_text("a file where the mirror folder should go")
    hook = U.EpochHook(run_dir, blocker / "r", sidecar, attempt)

    class Trainer:
        epoch, epochs = 0, 10

    import logging

    logging.disable(logging.WARNING)
    try:
        hook(Trainer())  # must not raise
    finally:
        logging.disable(logging.NOTSET)
    assert attempt["end_epoch"] == 1


def test_onnx_layout_classification():
    assert U._classify_layout([1, 300, 6], 3) == "e2e"
    assert U._classify_layout([1, 7, 8400], 3) == "raw"
    assert U._classify_layout([1, 5, 8400], 1) == "raw"
    assert U._classify_layout([1, 6, 8400], 2) == "raw"  # nc=2 raw also has a 6
    _raises(RuntimeError, lambda: U._classify_layout([1, 84, 8400], 3), "unexpected")
    _raises(RuntimeError, lambda: U._classify_layout([1, "N", 6], 3), "unexpected")


# ---------------------------------------------------------------- dispatch


def test_dispatch_routes_on_framework():
    d = _tmp()
    deim_cfg = d / "deim_s_def3_fold0.yaml"
    deim_cfg.write_text("name: deim_s_def3_fold0\nframework: deim\nadapter: deim\n"
                        "model: dfine_s\ndataset_id: ds_v2_def3\nfold: fold0\n")
    if not hasattr(__import__("train.deim_run", fromlist=["x"]), "run_from_config"):
        _raises(NotImplementedError, lambda: dispatch.run(deim_cfg), "run_from_config")
        _raises(NotImplementedError, lambda: dispatch.resolve(deim_cfg), "run_from_config")
    bad = d / "bad.yaml"
    bad.write_text("name: x\nframework: tensorflow\n")
    _raises(B.ConfigError, lambda: dispatch.run(bad), "unknown framework")
    assert dispatch.framework_of({}) == "ultralytics"
    eff = dispatch.resolve(ML / "configs" / "matrix" / "yolo26n_def3_fold0.yaml", {"epochs": 60})
    assert eff["epochs"] == 60 and eff["name"] == "yolo26n_def3_fold0"


def test_dispatch_cli_maps_errors_to_exit_codes():
    d = _tmp()
    bad = d / "typo.yaml"
    bad.write_text("name: t_def3_fold0\nmodel: yolo11n.pt\ndataset_id: ds_v2_def3\nepoch: 3\n")
    import io
    import logging
    from contextlib import redirect_stdout

    logging.disable(logging.CRITICAL)
    try:
        with redirect_stdout(io.StringIO()):
            assert dispatch.main(["--config", str(bad)]) == dispatch.EXIT_CONFIG
            assert dispatch.main(["--config", str(d / "missing.yaml")]) == dispatch.EXIT_CONFIG
            assert dispatch.main(["--config", str(ML / "configs" / "smoke_def3.yaml"),
                                  "--set", "name=x"]) == dispatch.EXIT_CONFIG
    finally:
        logging.disable(logging.NOTSET)


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
