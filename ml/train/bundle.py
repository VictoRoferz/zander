"""
Bundle v2 helpers shared by every training runner (Ultralytics today, DEIM next).

Pure functions: stdlib + PyYAML only, no torch / ultralytics import, so the
matrix driver, the tests and other runners can use them without a training
stack. The bundle format is docs/CONTRACTS.md §8.1; this module is its single
writer-side implementation.

Stable API (keyword names are part of the API; additions are keyword-only):

    ConfigError                                   invalid run config / override (ValueError)
    load_config(path) -> dict                     YAML mapping
    parse_overrides(["a.b=1", ...]) -> dict       {"a.b": 1}; values parsed as YAML scalars
    effective_config(cfg, overrides=None, *, defaults=None,
                     allowed_keys=None, locked_keys=LOCKED_KEYS) -> dict
                                                  defaults <- cfg <- overrides (deep merge)
    dump_config(cfg) -> str                       canonical YAML (sorted keys) = bundle config.yaml
    config_sha256(cfg) -> str                     sha256 of dump_config(cfg)
    parse_run_name(name) -> RunName | None        "<arch>_<schema>_fold<k>" / "..._devall"
    family_of(name) -> str                        "<arch>_<schema>" (or the name itself)
    arch_of(name) -> str
    sha256_file(path) -> str
    build_meta(*, config, provenance, framework, adapter, weights, epochs_completed,
               attempts, onnx, onnx_raw, versions, overrides=None, git=None,
               extra=None, created=None) -> dict   meta.json (all v1 + v2 keys)
    write_bundle(zip_path, files, *, required=REQUIRED_BUNDLE_FILES) -> (Path, sha256)
                                                  flat zip + "<zip>.sha256" marker, both atomic
    sha_path_for(zip_path) -> Path                "<zip>.sha256"
    read_sha256_file(path) -> str | None
    verify_bundle(zip_path) -> BundleCheck        marker present + sha matches + readable meta
                                                  (io_error=True: the check itself failed)
    read_bundle_meta(zip_path) -> dict
    atomic_write_bytes / atomic_write_text / atomic_copy / utc_now

A DEIM runner builds its bundle with the same calls: effective_config(...,
defaults=<its defaults>, allowed_keys=<its keys>) -> config_sha256 ->
build_meta(framework="deim", adapter="deim", onnx={... layout "deim"},
versions={"python", "torch", "cuda", "deim_commit"}, extra={...}) ->
write_bundle(<run>_bundle.zip, {"config.yaml": dump_config(eff), "meta.json": ...,
"LICENSE_DEIM": <path>, ...}).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Union

import yaml

BUNDLE_VERSION = 2

# Top-level keys a run config may carry (the common schema; a framework runner
# may pass its own `allowed_keys`). Unknown keys are rejected so that a typo
# ("epoch: 60") can never silently train with the default instead.
DEFAULT_ALLOWED_KEYS = frozenset(
    {
        "name", "framework", "adapter", "model", "dataset_id", "fold",
        "epochs", "imgsz", "batch", "seed", "deterministic", "final_weights",
        "augment", "extra", "export",
    }
)
# Identity keys: they define the run name, family and data; overriding them on
# the command line would silently mislabel a bundle. Copy the config instead.
LOCKED_KEYS = frozenset({"name", "framework", "adapter", "model", "dataset_id", "fold"})

REQUIRED_BUNDLE_FILES = ("config.yaml", "meta.json", "metrics.json", "model.onnx")
ONNX_LAYOUTS = ("raw", "e2e", "deim")
WEIGHTS_CHOICES = ("last", "best")

# meta.json keys (docs/CONTRACTS.md §8.1): every v1 key plus the v2 additions.
META_V1_KEYS = (
    "run_name", "framework", "adapter", "fold", "dataset_id", "schema", "classes",
    "dataset_fingerprint", "snapshot_id", "split_id", "tiler_version",
    "augmentation_offline", "augmentation_online", "ultralytics_version",
    "torch_version", "gpu", "git_sha", "git_dirty",
)
META_V2_KEYS = (
    "bundle_version", "family", "weights", "epochs", "epochs_completed",
    "train_seconds", "attempts", "config_sha256", "overrides", "onnx", "onnx_raw",
    "versions", "created",
)
ONNX_INFO_KEYS = ("file", "sha256", "opset", "layout", "output_shapes")

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RUN_RE = re.compile(
    r"^(?P<family>(?P<arch>.+)_(?P<schema>[A-Za-z0-9]+))_(?P<fold>fold\d+|devall)$"
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class ConfigError(ValueError):
    """The run config or an override is invalid (a user error: retrying won't help)."""


# ------------------------------------------------------------------ config


def load_config(path: Union[str, Path]) -> dict:
    """Read a run config YAML; it must be a mapping."""
    path = Path(path)
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"config not found: {path}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from None
    if not isinstance(doc, dict):
        raise ConfigError(f"{path}: a run config must be a YAML mapping")
    return doc


def parse_overrides(pairs: Optional[Iterable[str]]) -> dict:
    """["epochs=60", "extra.val=false"] -> {"epochs": 60, "extra.val": False}.

    Values are parsed as YAML scalars (ints, floats, booleans, null, lists);
    an empty value is null. Later pairs win over earlier ones.
    """
    out: dict = {}
    for pair in pairs or []:
        key, sep, raw = str(pair).partition("=")
        key = key.strip()
        if not sep or not key:
            raise ConfigError(f"override {pair!r} is not KEY=VALUE")
        try:
            value = yaml.safe_load(raw) if raw.strip() else None
        except yaml.YAMLError as e:
            raise ConfigError(f"override {pair!r}: value is not valid YAML: {e}") from None
        out[key] = value
    return out


def _deep_merge(base: dict, update: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in update.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def effective_config(
    cfg: Mapping,
    overrides: Optional[Mapping] = None,
    *,
    defaults: Optional[Mapping] = None,
    allowed_keys: Optional[Iterable[str]] = None,
    locked_keys: Iterable[str] = LOCKED_KEYS,
) -> dict:
    """The config a run actually trains with: defaults <- cfg <- overrides.

    - Nested mappings are deep-merged (a config's `augment: {scale: 0.3}` keeps
      the other default augment keys).
    - Override keys are dotted paths ("extra.val", "export.opset"); intermediate
      mappings are created. A plain key with a mapping value is deep-merged.
    - Unknown top-level keys are rejected, in the config and in the overrides.
    - `locked_keys` (identity: name, model, fold, ...) cannot be overridden.
    The result is what gets dumped as the bundle's config.yaml.
    """
    if not isinstance(cfg, Mapping):
        raise ConfigError("a run config must be a mapping")
    allowed = frozenset(allowed_keys) if allowed_keys is not None else DEFAULT_ALLOWED_KEYS
    locked = frozenset(locked_keys)
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise ConfigError(
            f"unknown top-level key(s) {unknown} in the config; allowed: {sorted(allowed)}"
        )
    eff = _deep_merge(dict(defaults or {}), cfg)
    for key, value in (overrides or {}).items():
        path = str(key).split(".")
        if any(not part for part in path):
            raise ConfigError(f"malformed override key {key!r}")
        top = path[0]
        if top not in allowed:
            raise ConfigError(
                f"override {key!r}: unknown top-level key {top!r}; allowed: {sorted(allowed)}"
            )
        if top in locked:
            raise ConfigError(
                f"override {key!r}: {top!r} is an identity key (it names the run/family); "
                "copy the config file instead of overriding it"
            )
        if len(path) == 1:
            if isinstance(value, Mapping) and isinstance(eff.get(top), dict):
                eff[top] = _deep_merge(eff[top], value)
            else:
                eff[top] = copy.deepcopy(value)
            continue
        node = eff.setdefault(top, {})
        for part in path[1:-1]:
            if not isinstance(node, dict):
                raise ConfigError(f"override {key!r}: {part!r} is not inside a mapping")
            node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override {key!r}: parent is not a mapping")
        node[path[-1]] = copy.deepcopy(value)
    name = eff.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ConfigError(
            f"config `name` must be a file-name-safe string ([A-Za-z0-9._-]), got {name!r}"
        )
    return eff


def dump_config(cfg: Mapping) -> str:
    """Canonical YAML: sorted keys, unicode kept. Deterministic, so its sha is
    the config's identity (config_sha256)."""
    return yaml.safe_dump(
        dict(cfg), sort_keys=True, allow_unicode=True, default_flow_style=False, width=100
    )


def config_sha256(cfg: Mapping) -> str:
    return hashlib.sha256(dump_config(cfg).encode("utf-8")).hexdigest()


# -------------------------------------------------------------- run names


class RunName(NamedTuple):
    family: str  # "<arch>_<schema>"
    arch: str    # e.g. "yolo26n", "deim_s"
    schema: str  # e.g. "def3", "nal1"
    fold: str    # "fold0".."foldK" or "dev_all" (the config's spelling)


def parse_run_name(name: str) -> Optional[RunName]:
    """Parse the matrix naming contract; None for free-form names (smoke, baseline)."""
    m = _RUN_RE.match(name or "")
    if not m:
        return None
    fold = "dev_all" if m["fold"] == "devall" else m["fold"]
    return RunName(m["family"], m["arch"], m["schema"], fold)


def family_of(name: str) -> str:
    """`yolo26n_def3_fold0` / `yolo26n_def3_devall` -> `yolo26n_def3`.

    A free-form name is its own family, so it can never be pooled with the
    matrix runs by accident."""
    parsed = parse_run_name(name)
    return parsed.family if parsed else name


def arch_of(name: str) -> str:
    """`yolo26s_def3_fold1` -> `yolo26s`; free-form names -> their first token."""
    parsed = parse_run_name(name)
    return parsed.arch if parsed else str(name).split("_", 1)[0]


# ------------------------------------------------------------------ files


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Union[str, Path], chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _fsync(f) -> None:
    f.flush()
    try:
        os.fsync(f.fileno())
    except OSError:  # some FUSE mounts refuse fsync; the rename is still atomic
        pass


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> Path:
    """temp file in the same folder -> fsync -> os.replace (never a torn file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            _fsync(f)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def atomic_write_text(path: Union[str, Path], text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_copy(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """Copy src to dst through a temp file in dst's folder + os.replace."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=dst.parent)
    try:
        with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
            for block in iter(lambda: inp.read(1 << 20), b""):
                out.write(block)
            _fsync(out)
        os.chmod(tmp, 0o644)
        os.replace(tmp, dst)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dst


# ------------------------------------------------------------------- meta


def build_meta(
    *,
    config: Mapping,
    provenance: Mapping,
    framework: str,
    adapter: str,
    weights: str,
    epochs_completed: Optional[int],
    attempts: list,
    onnx: Mapping,
    onnx_raw: Optional[Mapping],
    versions: Mapping,
    overrides: Optional[Mapping] = None,
    git: Optional[Mapping] = None,
    extra: Optional[Mapping] = None,
    created: Optional[str] = None,
) -> dict:
    """meta.json v2 = every v1 key + the v2 additions (docs/CONTRACTS.md §8.1).

    config      the EFFECTIVE config (its dump is the bundle's config.yaml)
    provenance  tiles/<dataset_id>/provenance.json of the training data
    attempts    [{gpu, started, seconds, resumed, ...}] — one per training
                process that advanced the run (resumes append)
    onnx        {file, sha256, opset, layout: raw|e2e|deim, output_shapes, ...}
    onnx_raw    same shape for model_raw.onnx (YOLO26 raw head) or None
    versions    {python, torch, cuda, ultralytics | deim_commit, ...}
    extra       framework-specific keys; may not overwrite a contract key
    """
    if weights not in WEIGHTS_CHOICES:
        raise ValueError(f"weights must be one of {WEIGHTS_CHOICES}, got {weights!r}")
    for label, info in (("onnx", onnx), ("onnx_raw", onnx_raw)):
        if info is None:
            if label == "onnx":
                raise ValueError("onnx info is required")
            continue
        missing = [k for k in ONNX_INFO_KEYS if k not in info]
        if missing:
            raise ValueError(f"{label} info lacks {missing}")
        if info["layout"] not in ONNX_LAYOUTS:
            raise ValueError(f"{label}.layout must be one of {ONNX_LAYOUTS}, got {info['layout']!r}")
    name = config["name"]
    attempts = [dict(a) for a in attempts]
    train_seconds = round(sum(float(a.get("seconds") or 0.0) for a in attempts), 1)
    git = dict(git or {})
    meta = {
        # ---- v1 keys ----
        "run_name": name,
        "framework": framework,
        "adapter": adapter,
        "fold": str(config.get("fold")),
        "dataset_id": config.get("dataset_id"),
        "schema": provenance.get("schema"),
        "classes": provenance.get("classes"),
        "dataset_fingerprint": provenance.get("dataset_fingerprint"),
        "snapshot_id": provenance.get("snapshot_id"),
        "split_id": provenance.get("split_id"),
        "tiler_version": provenance.get("tiler_version"),
        "augmentation_offline": provenance.get("augmentation"),
        "augmentation_online": config.get("augment"),
        "ultralytics_version": versions.get("ultralytics"),
        "torch_version": versions.get("torch"),
        "gpu": attempts[-1].get("gpu") if attempts else None,
        "git_sha": git.get("git_sha"),
        "git_dirty": git.get("git_dirty"),
        # ---- v2 keys ----
        "bundle_version": BUNDLE_VERSION,
        "family": family_of(name),
        "weights": weights,
        "epochs": config.get("epochs"),
        "epochs_completed": epochs_completed,
        "train_seconds": train_seconds,
        "attempts": attempts,
        "config_sha256": config_sha256(config),
        "overrides": dict(overrides or {}),
        "onnx": dict(onnx),
        "onnx_raw": dict(onnx_raw) if onnx_raw is not None else None,
        "versions": dict(versions),
        "created": created or utc_now(),
        # ---- convenience copies of the run-defining knobs ----
        "model": config.get("model"),
        "imgsz": config.get("imgsz"),
        "batch": config.get("batch"),
        "seed": config.get("seed"),
    }
    for k, v in (extra or {}).items():
        if k in meta:
            raise ValueError(f"extra meta key {k!r} collides with a contract key")
        meta[k] = v
    return meta


# ----------------------------------------------------------------- bundle


def sha_path_for(zip_path: Union[str, Path]) -> Path:
    zip_path = Path(zip_path)
    return zip_path.with_name(zip_path.name + ".sha256")


def read_sha256_file(path: Union[str, Path]) -> Optional[str]:
    """First token of a `sha256sum`-style line, or None if absent/malformed."""
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    token = text.split()[0].lower() if text else ""
    return token if _SHA_RE.match(token) else None


def _check_arcname(arc: str) -> None:
    if not arc or "/" in arc or "\\" in arc or arc.startswith(".") or arc in (".", ".."):
        raise ValueError(f"bundle member {arc!r} must be a bare file name (the zip is flat)")


def write_bundle(
    zip_path: Union[str, Path],
    files: Mapping[str, Union[str, bytes, Path]],
    *,
    required: Iterable[str] = REQUIRED_BUNDLE_FILES,
) -> tuple[Path, str]:
    """Write the flat bundle zip, then its `<zip>.sha256` marker (both atomic).

    files maps member name -> Path (copied), bytes, or str (UTF-8). The marker is
    written last: its presence means "this zip is complete". Returns
    (zip_path, sha256).
    """
    zip_path = Path(zip_path)
    for arc in files:
        _check_arcname(arc)
    missing = [r for r in required if r not in files]
    if missing:
        raise ValueError(f"bundle is missing required member(s) {missing}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{zip_path.name}.", suffix=".tmp", dir=zip_path.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for arc in sorted(files):
                src = files[arc]
                if isinstance(src, (bytes, str)):
                    zf.writestr(arc, src)
                else:
                    zf.write(Path(src), arc)
        with open(tmp, "rb+") as f:
            _fsync(f)
        os.chmod(tmp, 0o644)
        os.replace(tmp, zip_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    sha = sha256_file(zip_path)
    atomic_write_text(sha_path_for(zip_path), f"{sha}  {zip_path.name}\n")
    return zip_path, sha


@dataclass(frozen=True)
class BundleCheck:
    ok: bool
    reason: str
    sha256: Optional[str] = None
    meta: Optional[dict] = None
    # True when the check itself failed (Drive mount dropped, EIO, permission):
    # the bundle is UNKNOWN, not invalid — never move it aside or retrain over it.
    io_error: bool = False


def verify_bundle(zip_path: Union[str, Path]) -> BundleCheck:
    """A bundle is valid when its .sha256 marker exists and matches the zip, the
    zip is flat and readable, and it carries meta.json + config.yaml + model.onnx.

    An OSError while checking (other than "file not found") gives
    `io_error=True`: the content was never judged."""
    zip_path = Path(zip_path)
    try:
        if not zip_path.is_file():
            return BundleCheck(False, "missing")
        try:
            marker = sha_path_for(zip_path).read_text(encoding="utf-8").strip()
        except (FileNotFoundError, UnicodeDecodeError):  # absent / garbage = no marker
            marker = ""
        token = marker.split()[0].lower() if marker else ""
        expected = token if _SHA_RE.match(token) else None
        if expected is None:
            return BundleCheck(False, "no valid .sha256 marker (incomplete copy?)")
        actual = sha256_file(zip_path)
        if actual != expected:
            return BundleCheck(False, f"sha256 mismatch (marker {expected[:12]}, zip {actual[:12]})")
        # the sha matched, so the bytes are the ones the writer verified: any
        # OSError below is the medium, not the content
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                for n in names:
                    _check_arcname(n)
                for r in ("meta.json", "config.yaml", "model.onnx"):
                    if r not in names:
                        return BundleCheck(False, f"zip lacks {r}", actual)
                meta = json.loads(zf.read("meta.json").decode("utf-8"))
        except (zipfile.BadZipFile, ValueError, KeyError) as e:
            return BundleCheck(False, f"unreadable zip: {e}", actual)
    except OSError as e:
        return BundleCheck(False, f"I/O error while checking: {type(e).__name__}: {e}", io_error=True)
    return BundleCheck(True, "ok", actual, meta)


def read_bundle_meta(zip_path: Union[str, Path]) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read("meta.json").decode("utf-8"))


def to_jsonable(obj: Any) -> Any:
    """Paths/sets/tuples -> JSON-friendly values (for meta and logs)."""
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
