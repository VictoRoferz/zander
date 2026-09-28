"""
Tiler v2: annotated photos -> 640-px tiles + offline augmentation -> COCO
ground truth per training/validation view, for one class schema.

Output layout (gitignored, deterministic, fingerprinted):
  ml_data/tiles/<dataset_id>/              dataset_id = ds_v2_<schema>[<ablation suffix>]
    images/<tile>_d<k>[_ct<target8>].jpg   one file per distinct (tile, D4 element, recolour)
    coco/fold<i>_train.json                folds != i  + train_always   (augmented)
    coco/fold<i>_val.json                  fold i                       (never augmented)
    coco/dev_all_train.json                all folds   + train_always   (final retrain)
    coco/test.json                         test bucket (reference only — the honest
                                           number comes from the photo-level eval)
    provenance.json                        params, per-view stats, dataset_fingerprint

Rules (plan 2026-09-28; principle: augment only against MEASURED nuisance
variation, and symmetrically for positives and negatives):

  - geometry/clipping from infer.tiling (the ONE slicer); tiles re-encoded as
    JPEG q95 4:4:4 — 28-px defects and solder colour must survive.
  - every tile has a canonical D4 orientation k0 (seeded by its name) used in
    EVERY view, positives and negatives alike, so orientation carries no label
    information. The in-training validation monitor therefore sees rotated
    tiles too — harmless: the model-selection metrics come from the
    photo-level evaluator, which always runs on the original photo.
  - training views only:
      * partial-box policy: a tile in which a schema box is cut to
        0 < visibility < PARTIAL_MIN_VISIBILITY is dropped (the 128-px overlap
        guarantees the defect is whole in a neighbour tile; slivers would teach
        "defect = background").
      * probe exclusion: a tile touching a probe box (board-A/A2 Brücke/Fahne,
        see dataprep/split.py) is dropped if the schema trains that class — no
        model ever sees a board-A bridge/flag.
      * handheld cap: negative tiles from handheld photos are capped at the
        number of seated negative tiles (171 of 194 iO photos are handheld,
        not the production pose).
      * class-aware repeat-factor sampling (LVIS RFS, Gupta et al. 2019):
        r = max_c max(1, sqrt(t / f_c)) over the classes in a tile, f_c = the
        fraction of the view's tiles containing class c, t solved per view so
        positive tiles make up ~POS_SHARE_TARGET of the entries, r <= R_MAX.
        Stochastic rounding is seeded per tile.
      * repeat copies are distinct D4 elements (a seeded permutation), not
        pixel-identical duplicates.
      * cross-board colour transfer: each training entry is, with probability
        ct_p, recoloured (Reinhard LAB mean/std transfer, whole-photo stats of
        a downsampled source and target photo) to a random photo of the OTHER
        board type (A/A2 <-> B). Attacks the colour part of the class<->board
        shortcut ("this board's colour => bridge / no NAL"); board LAYOUT cues
        remain — only cross-product data can remove those. Targets are drawn
        from non-test photos.
  - unannotated, quarantined and review photos produce no tiles.
  - COCO category_id = schema class id + 1; the YOLO view maps it back.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from config.settings import settings
from dataprep.attributes import attributes_hash
from dataprep.ingest import load_manifest, photo_path, usable
from dataprep.remap import project, schema_classes
from dataprep.split import bucket_of, load_split, train_buckets
from infer.tiling import (
    MIN_SIDE,
    MIN_VISIBILITY,
    STRIDE,
    TILE_SIZE,
    TILER_VERSION,
    box_visibility,
    clip_box_to_tile,
    d4_box,
    d4_image,
    make_tile_name,
    tile_origins,
    xyxy_to_coco,
)

logger = logging.getLogger("ml.tiler")

DATASET_PREFIX = "ds_v2"
ENCODING = "jpeg95"
BOARD_TYPE = {"A": "A", "A2": "A", "B": "B"}  # colour-transfer domains


@dataclass(frozen=True)
class AugConfig:
    d4: bool = True
    ct_p: float = 0.3
    handheld_cap: bool = True
    rfs: bool = True
    pos_share_target: float = 0.30
    r_max: int = 8
    fixed_k: int = 7  # only used when rfs=False (the v1 policy, for ablation)
    partial_min_visibility: float = 0.5

    def suffix(self) -> str:
        d = AugConfig()
        parts = []
        if self.d4 != d.d4:
            parts.append("nod4")
        if self.ct_p != d.ct_p:
            parts.append(f"ct{int(round(self.ct_p * 100))}")
        if self.handheld_cap != d.handheld_cap:
            parts.append("nocap")
        if self.rfs != d.rfs:
            parts.append(f"k{self.fixed_k}")
        return "".join(f"_{p}" for p in parts)


DEFAULT_AUG = AugConfig()


def dataset_id(schema: str, aug: AugConfig = DEFAULT_AUG) -> str:
    return f"{DATASET_PREFIX}_{schema}{aug.suffix()}"


def _rng(*parts) -> random.Random:
    return random.Random("|".join(str(p) for p in (settings.seed, *parts)))


# ---------------------------------------------------------------- records


def _tile_records(manifest: dict, split: dict, schema: str) -> list[dict]:
    """One record per tile of every usable photo (bucket-independent facts)."""
    classes = schema_classes(schema)
    probe = {}
    for p in split["probe_boxes"]:
        if p["class"] in classes:
            probe.setdefault(p["capture_id"], []).append(tuple(p["box"]))
    records = []
    for e in manifest["entries"]:
        if not usable(e):
            continue
        train_boxes, _ = project(e["boxes"], schema)
        bucket = bucket_of(e, split)
        for ox, oy in tile_origins(e["width"], e["height"]):
            boxes, partial = [], False
            for cid, b in train_boxes:
                vis = box_visibility(b, (ox, oy))
                if 0 < vis < DEFAULT_AUG.partial_min_visibility:
                    partial = True
                clipped = clip_box_to_tile(b, (ox, oy))
                if clipped is not None:
                    boxes.append((cid, clipped))
            name = make_tile_name(e["capture_id"], ox, oy)
            records.append(
                {
                    "capture_id": e["capture_id"],
                    "name": name,
                    "origin": [ox, oy],
                    "board": e["board"],
                    "pose": e["pose"],
                    "bucket": bucket,
                    "verdict": e["verdict"],
                    "classes_present": e["classes_present"],
                    "boxes": boxes,
                    "partial": partial,
                    "probe": any(
                        box_visibility(b, (ox, oy)) > 0 for b in probe.get(e["capture_id"], [])
                    ),
                }
            )
    return records


def _perm(name: str, aug: AugConfig) -> list[int]:
    if not aug.d4:
        return [0] * 8
    p = list(range(8))
    _rng("d4", name).shuffle(p)
    return p


# ------------------------------------------------------------ view building


def _solve_repeat(recs: list[dict], n_classes: int, aug: AugConfig) -> tuple[float, list[float]]:
    """Return (t, r per record) for class-aware repeat-factor sampling."""
    n = len(recs)
    has = [set(c for c, _ in r["boxes"]) for r in recs]
    f = [sum(1 for h in has if c in h) / n if n else 0.0 for c in range(n_classes)]

    def factors(t: float) -> list[float]:
        out = []
        for h in has:
            if not h:
                out.append(1.0)
                continue
            r = max(max(1.0, math.sqrt(t / f[c])) for c in h if f[c] > 0)
            out.append(min(float(aug.r_max), r))
        return out

    def share(rs: list[float]) -> float:
        pos = sum(r for r, h in zip(rs, has) if h)
        neg = sum(1 for h in has if not h)
        return pos / (pos + neg) if pos + neg else 0.0

    if share(factors(0.0)) >= aug.pos_share_target:
        return 0.0, factors(0.0)
    # r = sqrt(t / f_c) reaches r_max for every class once t >= r_max^2 * max f_c
    lo, hi = 0.0, float(aug.r_max) ** 2 * max(f)
    if share(factors(hi)) < aug.pos_share_target:  # capped by r_max
        return hi, factors(hi)
    for _ in range(60):
        mid = (lo + hi) / 2
        if share(factors(mid)) < aug.pos_share_target:
            lo = mid
        else:
            hi = mid
    return hi, factors(hi)


def _ct_pool(manifest: dict, split: dict) -> dict[str, list[str]]:
    """Colour-transfer targets per board type: usable, non-test photos."""
    pool: dict[str, list[str]] = {"A": [], "B": []}
    for e in manifest["entries"]:
        if usable(e) and bucket_of(e, split) != "test" and e["board"] in BOARD_TYPE:
            pool[BOARD_TYPE[e["board"]]].append(e["capture_id"])
    return {k: sorted(v) for k, v in pool.items()}


def _file_name(name: str, k: int, ct: str | None) -> str:
    return f"{name}_d{k}" + (f"_ct{ct[:8]}" if ct else "") + ".jpg"


def _entries_for_view(
    recs: list[dict], train: bool, aug: AugConfig, n_classes: int, pool: dict
) -> tuple[list[dict], dict]:
    """Expand records into COCO image entries (with augmentation if train)."""
    stats = Counter()
    if train:
        kept = []
        for r in recs:
            if r["partial"]:
                stats["dropped_partial"] += 1
            elif r["probe"]:
                stats["dropped_probe"] += 1
            else:
                kept.append(r)
        if aug.handheld_cap:
            neg = [r for r in kept if not r["boxes"]]
            seated = sum(1 for r in neg if r["pose"] == "seated")
            handheld = sorted(
                (r for r in neg if r["pose"] == "handheld"),
                key=lambda r: hashlib.sha256(f"{settings.seed}|cap|{r['name']}".encode()).hexdigest(),
            )
            drop = {r["name"] for r in handheld[seated:]}
            stats["dropped_handheld_cap"] = len(drop)
            kept = [r for r in kept if r["name"] not in drop]
        recs = kept
        if aug.rfs:
            t, factors = _solve_repeat(recs, n_classes, aug)
        else:
            t, factors = None, [float(aug.fixed_k) if r["boxes"] else 1.0 for r in recs]
        stats["rfs_t"] = t
    else:
        factors = [1.0] * len(recs)

    entries = []
    for r, fac in zip(recs, factors):
        if train:
            base = int(math.floor(fac))
            copies = base + (1 if _rng("round", r["name"]).random() < fac - base else 0)
        else:
            copies = 1
        perm = _perm(r["name"], aug)
        other = "B" if BOARD_TYPE.get(r["board"]) == "A" else "A"
        for j in range(copies):
            k = perm[j % 8]
            ct = None
            if train and aug.ct_p > 0 and pool.get(other):
                rng = _rng("ct", r["name"], j)
                if rng.random() < aug.ct_p:
                    ct = pool[other][rng.randrange(len(pool[other]))]
            entries.append({"record": r, "k": k, "ct": ct, "file_name": _file_name(r["name"], k, ct)})
    return entries, dict(stats)


def _coco(entries: list[dict], classes: list[str]) -> dict:
    images, annotations = [], []
    ann_id = 0
    for img_id, en in enumerate(entries, 1):
        r = en["record"]
        images.append(
            {
                "id": img_id,
                "file_name": en["file_name"],
                "width": TILE_SIZE,
                "height": TILE_SIZE,
                "capture_id": r["capture_id"],
                "origin": r["origin"],
                "d4": en["k"],
                "ct_target": en["ct"],
                "board": r["board"],
                "pose": r["pose"],
                "bucket": r["bucket"],
                "photo_verdict": r["verdict"],
                "photo_classes": r["classes_present"],
            }
        )
        for cid, b in r["boxes"]:
            ann_id += 1
            coco_box = xyxy_to_coco(d4_box(b, en["k"]))
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid + 1,
                    "bbox": [round(v, 2) for v in coco_box],
                    "area": round(coco_box[2] * coco_box[3], 2),
                    "iscrowd": 0,
                }
            )
    categories = [{"id": i + 1, "name": c} for i, c in enumerate(classes)]
    return {"images": images, "annotations": annotations, "categories": categories}


def _view_stats(entries: list[dict], classes: list[str]) -> dict:
    pos = [e for e in entries if e["record"]["boxes"]]
    neg = [e for e in entries if not e["record"]["boxes"]]
    inst = Counter(classes[c] for e in entries for c, _ in e["record"]["boxes"])
    tiles_with = Counter(
        classes[c] for e in entries for c in {c for c, _ in e["record"]["boxes"]}
    )

    def orient(es):
        h = Counter(e["k"] for e in es)
        return [round(h.get(k, 0) / max(len(es), 1), 3) for k in range(8)]

    return {
        "n_entries": len(entries),
        "n_unique_files": len({e["file_name"] for e in entries}),
        "positive_share": round(len(pos) / max(len(entries), 1), 4),
        "instances": dict(inst),
        "entries_with_class_share": {
            c: round(tiles_with.get(c, 0) / max(len(entries), 1), 4) for c in classes
        },
        "orientation_pos": orient(pos),
        "orientation_neg": orient(neg),
        "ct_share_pos": round(sum(1 for e in pos if e["ct"]) / max(len(pos), 1), 3),
        "ct_share_neg": round(sum(1 for e in neg if e["ct"]) / max(len(neg), 1), 3),
        "photos": len({e["record"]["capture_id"] for e in entries}),
    }


def build_views(schema: str, aug: AugConfig = DEFAULT_AUG) -> tuple[dict, dict, list[dict]]:
    """Return ({view: coco}, provenance, all entries) without touching any image."""
    manifest, split = load_manifest(), load_split()
    classes = schema_classes(schema)
    records = _tile_records(manifest, split, schema)
    pool = _ct_pool(manifest, split)

    def recs_in(buckets: set[str]) -> list[dict]:
        return [r for r in records if r["bucket"] in buckets]

    view_specs = {}
    for i in sorted(split["folds"]):
        view_specs[f"fold{i}_train"] = (train_buckets(split, i), True)
        view_specs[f"fold{i}_val"] = ({f"fold{i}"}, False)
    view_specs["dev_all_train"] = (train_buckets(split, None), True)
    view_specs["test"] = ({"test"}, False)

    cocos, view_stats, all_entries = {}, {}, []
    for view, (buckets, train) in view_specs.items():
        entries, stats = _entries_for_view(recs_in(buckets), train, aug, len(classes), pool)
        cocos[view] = _coco(entries, classes)
        view_stats[view] = {**stats, **_view_stats(entries, classes)}
        all_entries.extend(entries)

    h = hashlib.sha256()
    h.update(
        f"{TILER_VERSION}|{TILE_SIZE}|{STRIDE}|{MIN_VISIBILITY}|{MIN_SIDE}|{ENCODING}|"
        f"{schema}|{manifest['snapshot_id']}|{split['split_id']}|{attributes_hash()}|"
        f"{json.dumps(asdict(aug), sort_keys=True)}|{settings.seed}".encode()
    )
    for view in sorted(cocos):
        h.update(view.encode())
        for im in cocos[view]["images"]:
            h.update(im["file_name"].encode())
        for a in cocos[view]["annotations"]:
            h.update(f"{a['image_id']},{a['category_id']},{a['bbox']}".encode())
    provenance = {
        "dataset_id": dataset_id(schema, aug),
        "schema": schema,
        "classes": classes,
        "tiler_version": TILER_VERSION,
        "tile_size": TILE_SIZE,
        "stride": STRIDE,
        "min_visibility": MIN_VISIBILITY,
        "min_side": MIN_SIDE,
        "encoding": ENCODING,
        "snapshot_id": manifest["snapshot_id"],
        "split_id": split["split_id"],
        "attributes_hash": attributes_hash(),
        "seed": settings.seed,
        "augmentation": asdict(aug),
        "n_photos_tiled": len({r["capture_id"] for r in records}),
        "n_base_tiles": len(records),
        "views": view_stats,
        "dataset_fingerprint": h.hexdigest(),
    }
    return cocos, provenance, all_entries


# ------------------------------------------------------------ image writing


def _lab_stats(path: Path):
    """Per-channel LAB mean/std of a photo (downsampled) for colour transfer."""
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        im.draft("RGB", (654, 490))
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return lab.mean(axis=0), lab.std(axis=0) + 1e-6


def colour_transfer(tile_rgb, src_stats, tgt_stats):
    """Reinhard transfer in LAB: match the source photo's stats to the target's."""
    import cv2
    import numpy as np

    arr = tile_rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    (ms, ss), (mt, st) = src_stats, tgt_stats
    lab = (lab - ms) / ss * st + mt
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def write_tile_images(entries: list[dict], out_dir: Path) -> int:
    import numpy as np
    from PIL import Image

    manifest = load_manifest()
    by_id = {e["capture_id"]: e for e in manifest["entries"]}
    todo: dict[str, dict[str, dict]] = {}
    for en in entries:
        if not (out_dir / en["file_name"]).exists():
            todo.setdefault(en["record"]["capture_id"], {})[en["file_name"]] = en
    stats_cache: dict[str, tuple] = {}

    def stats(cid: str):
        if cid not in stats_cache:
            stats_cache[cid] = _lab_stats(photo_path(by_id[cid]))
        return stats_cache[cid]

    written = 0
    for n, (cid, files) in enumerate(sorted(todo.items()), 1):
        with Image.open(photo_path(by_id[cid])) as im:
            im = im.convert("RGB")
            for fname, en in sorted(files.items()):
                ox, oy = en["record"]["origin"]
                tile = im.crop((ox, oy, ox + TILE_SIZE, oy + TILE_SIZE))
                if en["ct"]:
                    tile = Image.fromarray(
                        colour_transfer(np.asarray(tile), stats(cid), stats(en["ct"]))
                    )
                d4_image(tile, en["k"]).save(out_dir / fname, quality=95, subsampling=0)
                written += 1
        if n % 25 == 0:
            logger.info(f"tiled {n}/{len(todo)} photos ({written} files)")
    return written


def run(schema: str = "def3", write_images: bool = True, aug: AugConfig = DEFAULT_AUG) -> Path:
    cocos, provenance, entries = build_views(schema, aug)
    ds_dir = settings.tiles_dir / provenance["dataset_id"]
    (ds_dir / "images").mkdir(parents=True, exist_ok=True)
    (ds_dir / "coco").mkdir(parents=True, exist_ok=True)
    if write_images:
        n = write_tile_images(entries, ds_dir / "images")
        logger.info(f"wrote {n} new tile files")
    for name, doc in cocos.items():
        (ds_dir / "coco" / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")
        v = provenance["views"][name]
        logger.info(
            f"coco/{name}.json: {v['n_entries']} entries ({v['n_unique_files']} files), "
            f"positive share {v['positive_share']:.2f}, instances {v['instances']}"
        )
    (ds_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        f"dataset {provenance['dataset_id']}: {provenance['n_base_tiles']} base tiles "
        f"from {provenance['n_photos_tiled']} photos, "
        f"fingerprint={provenance['dataset_fingerprint'][:16]}…"
    )
    return ds_dir


def load_provenance(ds_id: str) -> dict:
    path = settings.tiles_dir / ds_id / "provenance.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python cli.py tile` first")
    return json.loads(path.read_text(encoding="utf-8"))
