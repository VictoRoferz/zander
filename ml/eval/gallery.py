"""
Single-file HTML error gallery: FN and FP crops with context, base64-embedded
(the repo gitignores *.jpg/*.png, so reports must be self-contained files).
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageDraw

CONTEXT_PX = 140
MAX_ITEMS_PER_SECTION = 60


def _crop_b64(photo: Image.Image, box, color: str) -> str:
    x1, y1, x2, y2 = box
    w, h = photo.size
    cx1, cy1 = max(0, int(x1) - CONTEXT_PX), max(0, int(y1) - CONTEXT_PX)
    cx2, cy2 = min(w, int(x2) + CONTEXT_PX), min(h, int(y2) + CONTEXT_PX)
    crop = photo.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    draw = ImageDraw.Draw(crop)
    draw.rectangle(
        [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], outline=color, width=3
    )
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def write_gallery(
    out_path: Path,
    source_dir: Path,
    fn_items: list[dict],  # {capture_id, box}
    fp_items: list[dict],  # {capture_id, box, score, other_defect: bool}
    title: str,
) -> Path:
    def section(name: str, items: list[str]) -> str:
        body = "\n".join(items) if items else "<p><em>none 🎉</em></p>"
        return f"<h2>{name} ({len(items)})</h2>\n<div class='grid'>{body}</div>"

    photo_cache: dict[str, Image.Image] = {}

    def photo(cid: str) -> Image.Image:
        if cid not in photo_cache:
            photo_cache[cid] = Image.open(source_dir / f"{cid}.jpg").convert("RGB")
        return photo_cache[cid]

    fn_cards = []
    for item in fn_items[:MAX_ITEMS_PER_SECTION]:
        b64 = _crop_b64(photo(item["capture_id"]), item["box"], "#e53935")
        fn_cards.append(
            f"<figure><img src='data:image/jpeg;base64,{b64}'>"
            f"<figcaption>FN · {item['capture_id'][:8]}</figcaption></figure>"
        )
    fp_cards = []
    for item in fp_items[:MAX_ITEMS_PER_SECTION]:
        tag = " · other-defect" if item.get("other_defect") else ""
        b64 = _crop_b64(photo(item["capture_id"]), item["box"], "#fb8c00")
        fp_cards.append(
            f"<figure><img src='data:image/jpeg;base64,{b64}'>"
            f"<figcaption>FP {item['score']:.2f}{tag} · "
            f"{item['capture_id'][:8]}</figcaption></figure>"
        )

    html = f"""<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
 .grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
 figure {{ margin: 0; border: 1px solid #ccc; padding: 4px; }}
 img {{ max-width: 280px; display: block; }}
 figcaption {{ font-size: 12px; color: #555; padding-top: 3px; }}
</style>
<h1>{title}</h1>
{section("False negatives (missed defects)", fn_cards)}
{section("False positives at operating point", fp_cards)}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    for im in photo_cache.values():
        im.close()
    return out_path
