"""
Pinned tile geometry — the single slicer used by dataset building, the
evaluation harness and the inspector service. Torch-free (pure python).

TILER v1 contract (changing any constant bumps TILER_VERSION and invalidates
every tile cache and metric comparison):
  - tile 640x640, stride 512 (20% overlap)
  - edge tiles are CLAMPED to the image edge (never zero-padded), so every
    tile is exactly 640x640 and no letterboxing exists anywhere in the system
  - offsets are a pure function of (width, height): for 2616x1960 they are
    x in {0, 512, 1024, 1536, 1976} and y in {0, 512, 1024, 1320} = 20 tiles
  - a tile's identity is its origin, encoded in the tile name
    "<capture_id>_x<xxxx>_y<yyyy>" so stitching needs no side lookup
  - ground-truth boxes are clipped to the tile; a clipped box is kept iff
    clipped_area / original_area >= MIN_VISIBILITY and both sides stay
    >= MIN_SIDE px. (Overlap 128 px > the 28 px median box, so every box with
    side <= 128 px is fully contained in at least one tile — clipping only
    affects duplicate partial copies in neighbouring tiles.)

Canonical box format everywhere: pixel [x1, y1, x2, y2] floats in the parent
image's coordinate system; rounding only happens at final file write.
"""
from __future__ import annotations

TILER_VERSION = "v1"
TILE_SIZE = 640
STRIDE = 512
MIN_VISIBILITY = 0.2
MIN_SIDE = 2.0


def axis_offsets(length: int, tile: int = TILE_SIZE, stride: int = STRIDE) -> list[int]:
    """Offsets along one axis; the last tile is clamped to the image edge."""
    if length <= tile:
        return [0]
    offs = list(range(0, length - tile + 1, stride))
    last = length - tile
    if offs[-1] != last:
        offs.append(last)
    return offs


def tile_origins(width: int, height: int) -> list[tuple[int, int]]:
    """All (x, y) tile origins for an image, row-major, deterministic."""
    return [
        (x, y)
        for y in axis_offsets(height)
        for x in axis_offsets(width)
    ]


def make_tile_name(capture_id: str, x: int, y: int) -> str:
    return f"{capture_id}_x{x:04d}_y{y:04d}"


def parse_tile_name(name: str) -> tuple[str, int, int]:
    """Inverse of make_tile_name; accepts names with or without extension."""
    stem = name.rsplit(".", 1)[0]
    base, xpart, ypart = stem.rsplit("_", 2)
    if not (xpart.startswith("x") and ypart.startswith("y")):
        raise ValueError(f"not a tile name: {name}")
    return base, int(xpart[1:]), int(ypart[1:])


def clip_box_to_tile(
    box: tuple[float, float, float, float],
    origin: tuple[int, int],
    tile: int = TILE_SIZE,
    min_visibility: float = MIN_VISIBILITY,
    min_side: float = MIN_SIDE,
) -> tuple[float, float, float, float] | None:
    """
    Clip a photo-coordinate xyxy box to one tile.

    Returns the clipped box in TILE coordinates, or None if the visible part
    is below the keep thresholds.
    """
    x1, y1, x2, y2 = box
    ox, oy = origin
    cx1, cy1 = max(x1, ox), max(y1, oy)
    cx2, cy2 = min(x2, ox + tile), min(y2, oy + tile)
    w, h = cx2 - cx1, cy2 - cy1
    if w <= 0 or h <= 0:
        return None
    orig_area = max((x2 - x1) * (y2 - y1), 1e-9)
    if (w * h) / orig_area < min_visibility:
        return None
    if w < min_side or h < min_side:
        return None
    return (cx1 - ox, cy1 - oy, cx2 - ox, cy2 - oy)


def tile_to_photo(
    box: tuple[float, float, float, float], origin: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Translate a tile-coordinate xyxy box back into photo coordinates."""
    ox, oy = origin
    x1, y1, x2, y2 = box
    return (x1 + ox, y1 + oy, x2 + ox, y2 + oy)


# ---- format converters (single code path for every export) ----


def xyxy_to_yolo(
    box: tuple[float, float, float, float], w: int, h: int
) -> tuple[float, float, float, float]:
    """Pixel xyxy -> YOLO normalized (cx, cy, bw, bh), clamped to [0, 1]."""
    x1, y1, x2, y2 = box
    cx = min(max((x1 + x2) / 2.0 / w, 0.0), 1.0)
    cy = min(max((y1 + y2) / 2.0 / h, 0.0), 1.0)
    bw = min(max((x2 - x1) / w, 0.0), 1.0)
    bh = min(max((y2 - y1) / h, 0.0), 1.0)
    return (cx, cy, bw, bh)


def yolo_to_xyxy(
    ybox: tuple[float, float, float, float], w: int, h: int
) -> tuple[float, float, float, float]:
    """YOLO normalized (cx, cy, bw, bh) -> pixel xyxy."""
    cx, cy, bw, bh = ybox
    return (
        (cx - bw / 2.0) * w,
        (cy - bh / 2.0) * h,
        (cx + bw / 2.0) * w,
        (cy + bh / 2.0) * h,
    )


def xyxy_to_coco(box: tuple[float, float, float, float]) -> list[float]:
    """Pixel xyxy -> COCO [x, y, w, h]."""
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]
