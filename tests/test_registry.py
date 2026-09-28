"""
Run registry v2 (ml/registry.py): bundle zip -> ml_data/bundles/<run>/ with no
temp-dir leak, MLflow experiment zander-v2 with the contract tags, idempotent
re-import, conflict refusal / --force archive, unsafe zips, the Drive .sha256
marker, bundle v1 compatibility, and log_eval.

Run: ml/.venv/bin/python -B tests/test_registry.py
"""
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="zander_registry_"))
os.environ["ML_DATA_ROOT"] = str(TMP)  # BEFORE config.settings is imported
sys.path.insert(0, str(ROOT / "ml"))

import registry  # noqa: E402
from config.settings import settings  # noqa: E402

NAMES = ["Nicht ausreichend Lot", "Brücke", "Fahne"]
SYS_TMP = Path(tempfile.gettempdir())


def _bundle_zip(path: Path, run_name: str, model: bytes = b"onnx-bytes-1", v2: bool = True,
                extra: dict | None = None) -> Path:
    meta = {"run_name": run_name, "framework": "ultralytics", "adapter": "ultralytics",
            "fold": run_name.rsplit("_", 1)[-1], "dataset_id": "ds_v2_def3", "schema": "def3",
            "classes": NAMES, "split_id": "split_v2", "tiler_version": "v2",
            "dataset_fingerprint": "f" * 64, "git_sha": "abc"}
    if v2:
        meta.update({"bundle_version": 2, "family": run_name.rsplit("_", 1)[0], "weights": "last",
                     "onnx": {"file": "model.onnx", "layout": "raw", "opset": 17}})
    config = f"name: {run_name}\nmodel: yolo26n.pt\ndataset_id: ds_v2_def3\nepochs: 120\nimgsz: 640\nbatch: 16\n"
    metrics = {"epochs": [{"epoch": 1, "train/box_loss": 1.25, "metrics/mAP50(B)": 0.1}]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("meta.json", json.dumps(meta))
        zf.writestr("config.yaml", config)
        zf.writestr("metrics.json", json.dumps(metrics))
        zf.writestr("model.onnx", model)
        zf.writestr("last.pt", b"weights")
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return path


def _sys_bundle_dirs() -> set:
    return {p.name for p in SYS_TMP.glob("bundle_*")}


def _runs(**tags):
    import mlflow

    mlflow.set_tracking_uri(f"file:{settings.mlruns_dir}")
    clause = " and ".join(f"tags.{k} = '{v}'" for k, v in tags.items())
    return mlflow.search_runs(experiment_names=[registry.EXPERIMENT], filter_string=clause,
                              output_format="list")


def _raises(fn, exc, *needles):
    try:
        fn()
    except exc as e:
        for n in needles:
            assert n in str(e), f"{n!r} not in: {e}"
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_import_extracts_into_bundles_without_temp_leak():
    before = _sys_bundle_dirs()
    z = _bundle_zip(TMP / "in" / "yolo26n_def3_fold0_bundle.zip", "yolo26n_def3_fold0")
    res = registry.import_bundle(z)
    target = settings.bundles_dir / "yolo26n_def3_fold0"
    assert res["status"] == "imported" and res["bundle_dir"] == target
    assert {p.name for p in target.iterdir()} >= {"model.onnx", "meta.json", "config.yaml", "last.pt"}
    assert (target / "model.onnx").read_bytes() == b"onnx-bytes-1"
    assert not [p for p in settings.bundles_dir.iterdir() if p.name.startswith(".staging")]
    assert _sys_bundle_dirs() == before, "import must not leave bundle_* dirs in the system temp"
    info = res["info"]
    assert (info["family"], info["fold"], info["schema"], info["bundle_version"]) == \
        ("yolo26n_def3", "fold0", "def3", 2)
    assert info["bundle_sha256"] == registry._bundle_sha(z)


def test_mlflow_experiment_and_tags():
    import mlflow

    mlflow.set_tracking_uri(f"file:{settings.mlruns_dir}")
    assert mlflow.get_experiment_by_name("zander-v2") is not None
    runs = _runs(run_name="yolo26n_def3_fold0")
    assert len(runs) == 1
    tags = runs[0].data.tags
    for key, value in (("family", "yolo26n_def3"), ("schema", "def3"), ("fold", "fold0"),
                       ("split_id", "split_v2"), ("benchmark", "split_v2"), ("model", "yolo26n.pt")):
        assert tags.get(key) == value, (key, tags.get(key))
    model_sha = registry.file_sha256(settings.bundles_dir / "yolo26n_def3_fold0" / "model.onnx")
    assert tags["model_sha256"] == model_sha and len(tags["bundle_sha"]) == 64
    assert runs[0].data.params["epochs"] == "120"
    art = {a.path for a in mlflow.MlflowClient().list_artifacts(runs[0].info.run_id)}
    assert "meta.json" in art and "model.onnx" not in art and "last.pt" not in art, art


def test_reimport_is_idempotent():
    z = TMP / "in" / "yolo26n_def3_fold0_bundle.zip"
    first = _runs(run_name="yolo26n_def3_fold0")[0].info.run_id
    res = registry.import_bundle(z)
    assert res["status"] == "present" and res["run_id"] == first
    assert registry.import_run(z) == first
    assert len(_runs(run_name="yolo26n_def3_fold0")) == 1


def test_different_bundle_same_run_name_needs_force():
    z2 = _bundle_zip(TMP / "in2" / "yolo26n_def3_fold0_bundle.zip", "yolo26n_def3_fold0",
                     model=b"onnx-bytes-2 retrained")
    _raises(lambda: registry.import_bundle(z2), registry.BundleError, "--force")
    target = settings.bundles_dir / "yolo26n_def3_fold0"
    assert (target / "model.onnx").read_bytes() == b"onnx-bytes-1", "refusal must not touch the old dir"
    res = registry.import_bundle(z2, force=True)
    assert res["status"] == "replaced" and (target / "model.onnx").read_bytes() == b"onnx-bytes-2 retrained"
    archived = list((settings.bundles_dir / registry.ARCHIVE_DIR).glob("yolo26n_def3_fold0__*"))
    assert len(archived) == 1 and (archived[0] / "model.onnx").read_bytes() == b"onnx-bytes-1"
    runs = _runs(run_name="yolo26n_def3_fold0")
    assert len(runs) == 2
    assert sum(1 for r in runs if "superseded_by" in r.data.tags) == 1
    new_sha = registry.file_sha256(target / "model.onnx")
    assert registry.find_mlflow_run(new_sha, None).info.run_id == res["run_id"]


def test_tampered_bundle_dir_is_not_silently_reused():
    z = _bundle_zip(TMP / "in" / "t_def3_fold1_bundle.zip", "t_def3_fold1")
    registry.import_bundle(z, register=False)
    (settings.bundles_dir / "t_def3_fold1" / "model.onnx").write_bytes(b"edited by hand")
    _raises(lambda: registry.import_bundle(z, register=False), registry.BundleError, "changed")


def test_unsafe_or_incomplete_zips_refused():
    bad = _bundle_zip(TMP / "in" / "evil_bundle.zip", "evil_def3_fold0", extra={"../escape.txt": "x"})
    _raises(lambda: registry.import_bundle(bad, register=False), registry.BundleError, "unsafe")
    nested = _bundle_zip(TMP / "in" / "nested_bundle.zip", "nested_def3_fold0", extra={"sub/x.txt": "x"})
    _raises(lambda: registry.import_bundle(nested, register=False), registry.BundleError, "flat")
    empty = TMP / "in" / "empty_bundle.zip"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("config.yaml", "name: x\n")
    _raises(lambda: registry.import_bundle(empty, register=False), registry.BundleError, "missing")
    assert not (settings.bundles_dir / "evil_def3_fold0").exists()
    assert not (TMP / "escape.txt").exists() and not (settings.bundles_dir / "escape.txt").exists()


def test_drive_sha_marker_is_verified():
    z = _bundle_zip(TMP / "drive" / "m_def3_fold2_bundle.zip", "m_def3_fold2")
    Path(str(z) + ".sha256").write_text("0" * 64 + "  m_def3_fold2_bundle.zip\n", encoding="utf-8")
    _raises(lambda: registry.import_bundle(z, register=False), registry.BundleError, "corrupt")
    Path(str(z) + ".sha256").write_text(registry._bundle_sha(z) + "\n", encoding="utf-8")
    assert registry.import_bundle(z, register=False)["status"] == "imported"


def test_bundle_v1_is_read():
    z = _bundle_zip(TMP / "in" / "yolo11s_def3_fold1_bundle.zip", "yolo11s_def3_fold1", v2=False)
    info = registry.import_bundle(z, register=False)["info"]
    assert info["bundle_version"] == 1 and info["family"] == "yolo11s_def3"
    assert info["weights"] == "last" and info["onnx_layout"] is None
    assert registry.family_from_run_name("yolo26n_def3_devall") == "yolo26n_def3"
    assert registry.family_from_run_name("yolo11n_smoke_def3") == "yolo11n_smoke_def3"


def test_bundle_dir_placed_by_hand_is_adopted_in_place():
    d = settings.bundles_dir / "hand_def3_fold1"
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"hand-made")
    (d / "meta.json").write_text(json.dumps({"run_name": "hand_def3_fold1", "fold": "fold1",
                                             "schema": "def3", "split_id": "split_v2"}))
    first = registry.import_bundle(d, register=False)
    assert first["status"] == "adopted" and first["bundle_dir"] == d
    assert (d / registry.BUNDLE_MARKER).is_file() and (d / "model.onnx").read_bytes() == b"hand-made"
    again = registry.import_bundle(d, register=False)
    assert again["status"] == "registered-in-place" and again["bundle_sha256"] == first["bundle_sha256"]


def test_log_eval_into_the_bundles_run():
    import mlflow

    target = settings.bundles_dir / "yolo26n_def3_fold0"
    sha = registry.file_sha256(target / "model.onnx")
    run_dir = settings.eval_dir / "yolo26n_def3_fold0"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"bucket": "fold0", "mean_ms_per_photo": 512.5,
               "source": {"model_sha256": sha, "bundle_run": "yolo26n_def3_fold0"},
               "level1_verdict": {"photo_auroc": 0.8, "escape": {"rate": 0.05},
                                  "false_call_seated": {"rate": 0.2}},
               "level3_defects": {"per_class": {"Nicht ausreichend Lot": {
                   "n_gt": 10, "recall_at_op": 0.7, "ap50": 0.6}}}}
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "report.md").write_text("# report\n", encoding="utf-8")
    run_id = registry.log_eval("yolo26n_def3_fold0")
    assert run_id is not None
    got = mlflow.get_run(run_id).data
    assert got.metrics["eval.fold0.recall_nal_0.5fppi"] == 0.7
    assert got.metrics["eval.fold0.ms_per_photo"] == 512.5 and got.metrics["eval.fold0.escape"] == 0.05
    assert got.tags["eval.fold0.run"] == "yolo26n_def3_fold0"
    art = {a.path for a in mlflow.MlflowClient().list_artifacts(run_id, "eval/yolo26n_def3_fold0")}
    assert art == {"eval/yolo26n_def3_fold0/metrics.json", "eval/yolo26n_def3_fold0/report.md"}, art
    # an eval of a never-imported model is reported, not attached to a random run
    other = settings.eval_dir / "unknown_run"
    other.mkdir()
    (other / "metrics.json").write_text(json.dumps({**metrics, "source": {"model_sha256": "9" * 64}}),
                                        encoding="utf-8")
    assert registry.log_eval("unknown_run") is None


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
