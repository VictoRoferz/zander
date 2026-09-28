"""
Unit tests for the pinned tile geometry + cross-tile merge (ml/infer).

The tiler is the single most correctness-critical code in the ML toolchain:
dataset GT, evaluation and the inspector all share it. These tests pin the
exact TILER contract (v1 grid, v2 D4 transforms) and — when the dataset has
been built locally — the invariants of the ds_v2 training views.

Run (no pytest needed):  python tests/test_tiler.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from infer.stitch import merge_detections  # noqa: E402
from infer.tiling import (  # noqa: E402
    TILE_SIZE,
    axis_offsets,
    clip_box_to_tile,
    make_tile_name,
    parse_tile_name,
    tile_origins,
    tile_to_photo,
    xyxy_to_yolo,
    yolo_to_xyxy,
)


def test_pinned_grid_for_native_resolution():
    # The exact offsets promised by the plan for 2616x1960.
    assert axis_offsets(2616) == [0, 512, 1024, 1536, 1976]
    assert axis_offsets(1960) == [0, 512, 1024, 1320]
    assert len(tile_origins(2616, 1960)) == 20


def test_grid_edge_cases():
    assert axis_offsets(640) == [0]          # exact fit
    assert axis_offsets(500) == [0]          # smaller than a tile
    assert axis_offsets(1152) == [0, 512]    # last stride lands exactly
    # Clamped final offset never duplicates and always reaches the edge.
    for length in (641, 1000, 1975, 2616, 4000):
        offs = axis_offsets(length)
        assert offs[-1] == length - TILE_SIZE
        assert len(offs) == len(set(offs))


def test_tile_name_round_trip():
    name = make_tile_name("abc123", 1976, 512)
    assert name == "abc123_x1976_y0512"
    assert parse_tile_name(name + ".jpg") == ("abc123", 1976, 512)


def test_coordinate_round_trip_under_half_pixel():
    # photo -> tile -> YOLO -> tile -> photo must stay < 0.51 px off.
    photo_box = (1540.3, 700.7, 1568.9, 731.2)  # ~29px box crossing x=1536 tile
    origin = (1536, 512)
    tile_box = clip_box_to_tile(photo_box, origin)
    assert tile_box is not None
    ybox = xyxy_to_yolo(tile_box, TILE_SIZE, TILE_SIZE)
    tile_back = yolo_to_xyxy(ybox, TILE_SIZE, TILE_SIZE)
    photo_back = tile_to_photo(tile_back, origin)
    for a, b in zip(photo_box, photo_back):
        assert abs(a - b) < 0.51, f"round-trip drift {abs(a - b)}"


def test_clipping_rules():
    origin = (0, 0)
    # fully inside -> unchanged
    assert clip_box_to_tile((10, 10, 40, 40), origin) == (10, 10, 40, 40)
    # 10% visible -> dropped (min visibility 0.2)
    assert clip_box_to_tile((636, 10, 676, 20), origin) is None
    # half visible -> kept, clipped to the tile edge
    clipped = clip_box_to_tile((620, 10, 660, 50), origin)
    assert clipped == (620, 10, 640, 50)
    # degenerate sliver (<2px side after clipping) -> dropped
    assert clip_box_to_tile((639, 100, 680, 140), origin) is None


def test_small_boxes_fully_contained_somewhere():
    # Plan guarantee: overlap (128px) > box side => every box <= 128px is
    # fully inside at least one tile. Sweep a 28px box across the image.
    for cx in range(14, 2616 - 14, 97):
        for cy in range(14, 1960 - 14, 97):
            box = (cx - 14, cy - 14, cx + 14, cy + 14)
            fully = False
            for origin in tile_origins(2616, 1960):
                clipped = clip_box_to_tile(box, origin, min_visibility=0.999)
                if clipped is not None:
                    fully = True
                    break
            assert fully, f"28px box at ({cx},{cy}) not fully contained anywhere"


def test_merge_fragments_of_large_box():
    # A ~700px defect predicted as two overlapping fragments from adjacent
    # tiles: IoU is low (0.38) but IoS is high -> must merge to the union.
    frag_a = (1000.0, 500.0, 1640.0, 560.0)   # left fragment
    frag_b = (1400.0, 500.0, 1700.0, 560.0)   # right fragment, IoS vs a = 0.8
    boxes, scores = merge_detections([frag_a, frag_b], [0.9, 0.7])
    assert len(boxes) == 1
    assert boxes[0] == (1000.0, 500.0, 1700.0, 560.0)  # union rectangle
    assert scores[0] == 0.9  # max score survives


def test_merge_keeps_distinct_detections():
    a = (100.0, 100.0, 130.0, 130.0)
    b = (400.0, 400.0, 430.0, 430.0)
    boxes, scores = merge_detections([a, b], [0.8, 0.6])
    assert len(boxes) == 2


def test_merge_duplicate_from_overlap_region():
    # The same 28px defect seen by two overlapping tiles -> near-identical
    # boxes -> exactly one survives.
    a = (1540.0, 700.0, 1568.0, 728.0)
    b = (1540.4, 700.2, 1568.5, 728.6)
    boxes, scores = merge_detections([a, b], [0.85, 0.83])
    assert len(boxes) == 1



# ---------------------------------------------------------------- TILER v2


def _inverse(k):
    r, f = k % 4, k // 4
    return k if f else (4 - r) % 4


def test_d4_box_matches_image_transform_for_all_8_elements():
    import numpy as np
    from PIL import Image

    from infer.tiling import d4_box, d4_image

    size = 64
    a = np.zeros((size, size), np.uint8)
    a[5:12, 20:40] = 255  # box x 20..40, y 5..12
    for k in range(8):
        b = np.asarray(d4_image(Image.fromarray(a), k))
        ref = np.rot90(np.fliplr(a) if k // 4 else a, k % 4)
        assert (b == ref).all(), f"PIL vs numpy convention differs at k={k}"
        ys, xs = np.nonzero(b)
        got = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        assert got == tuple(int(v) for v in d4_box((20, 5, 40, 12), k, size)), k


def test_d4_round_trip_and_area_preserved():
    from infer.tiling import d4_box

    box = (100.5, 30.25, 128.0, 71.0)
    area = (box[2] - box[0]) * (box[3] - box[1])
    for k in range(8):
        t = d4_box(box, k)
        assert abs((t[2] - t[0]) * (t[3] - t[1]) - area) < 1e-9  # no box inflation
        back = d4_box(t, _inverse(k))
        assert all(abs(u - v) < 1e-9 for u, v in zip(back, box)), k


def test_box_visibility():
    from infer.tiling import box_visibility

    assert box_visibility((10, 10, 40, 40), (0, 0)) == 1.0
    assert box_visibility((630, 10, 650, 30), (0, 0)) == 0.5
    assert box_visibility((700, 10, 720, 30), (0, 0)) == 0.0


def _built(schema):
    from config.settings import settings
    from dataprep.tiler import dataset_id

    d = settings.tiles_dir / dataset_id(schema)
    return d if (d / "provenance.json").exists() else None


def test_training_views_invariants():
    import json

    from dataprep.ingest import load_manifest, usable
    from dataprep.split import load_split

    manifest, split = load_manifest(), load_split()
    by_id = {e["capture_id"]: e for e in manifest["entries"]}
    probe_ids = {p["capture_id"] for p in split["probe_boxes"]}
    checked = 0
    for schema in ("def3", "nal1"):
        d = _built(schema)
        if d is None:
            continue
        prov = json.loads((d / "provenance.json").read_text(encoding="utf-8"))
        for view in ("fold0_train", "fold1_train", "fold2_train", "dev_all_train"):
            coco = json.loads((d / "coco" / f"{view}.json").read_text(encoding="utf-8"))
            assert [c["name"] for c in coco["categories"]] == prov["classes"]
            held_out = {"test"} | ({f"fold{view[4]}"} if view != "dev_all_train" else set())
            for im in coco["images"]:
                e = by_id[im["capture_id"]]
                assert usable(e), f"{view}: unusable capture {e['capture_id']}"
                assert im["bucket"] not in held_out, f"{view}: leaked {im['bucket']} tile"
            v = prov["views"][view]
            assert abs(v["positive_share"] - 0.30) < 0.02, (view, v["positive_share"])
            parity = max(abs(a - b) for a, b in zip(v["orientation_pos"], v["orientation_neg"]))
            assert parity < 0.06, f"{view}: orientation leaks label info ({parity:.3f})"
            if schema == "def3":
                # no training tile may show a probe box (board-A Brücke/Fahne)
                from infer.tiling import TILE_SIZE, box_visibility

                probe = {}
                for p in split["probe_boxes"]:
                    probe.setdefault(p["capture_id"], []).append(p["box"])
                for im in coco["images"]:
                    for b in probe.get(im["capture_id"], []):
                        assert box_visibility(b, tuple(im["origin"]), TILE_SIZE) == 0.0
            checked += 1
        for im in json.loads((d / "coco" / "fold0_val.json").read_text(encoding="utf-8"))["images"]:
            assert im["ct_target"] is None, "validation tiles must not be recoloured"
    if not checked:
        print("  (skipped: no ds_v2 dataset built locally)")


def test_view_building_is_deterministic():
    if _built("nal1") is None:
        print("  (skipped: no ds_v2 dataset built locally)")
        return
    from dataprep.tiler import build_views

    _, p1, _ = build_views("nal1")
    _, p2, _ = build_views("nal1")
    assert p1["dataset_fingerprint"] == p2["dataset_fingerprint"]


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
