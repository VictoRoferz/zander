"""
QA gallery of augmented training tiles: eyeball once per dataset build that
the boxes still sit on the defects after D4 rotation/flip and colour transfer.

    python cli.py qa --schema def3      # -> ml_data/tiles/<dataset_id>/qa.html

Samples ~2/3 positive and ~1/3 negative entries from dev_all_train (seeded),
draws the COCO boxes with class colours, and embeds the tiles as JPEG data URIs.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import random
from pathlib import Path

from PIL import Image, ImageDraw

from config.settings import settings
from dataprep.remap import SHORT
from dataprep.tiler import dataset_id

logger = logging.getLogger("ml.qa")

COLOURS = ["#e8412c", "#2a78d6", "#f2b400"]  # NAL, Brücke, Fahne


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def run(schema: str = "def3", n: int = 48) -> Path:
    ds_id = dataset_id(schema)
    ds_dir = settings.tiles_dir / ds_id
    coco = json.loads((ds_dir / "coco" / "dev_all_train.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns: dict[int, list] = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    rng = random.Random(settings.seed)
    pos = [im for im in coco["images"] if im["id"] in anns]
    neg = [im for im in coco["images"] if im["id"] not in anns]
    sample = rng.sample(pos, min(len(pos), 2 * n // 3)) + rng.sample(neg, min(len(neg), n // 3))

    cards = []
    for im in sample:
        tile = Image.open(ds_dir / "images" / im["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(tile)
        labels = []
        for a in anns.get(im["id"], []):
            x, y, w, h = a["bbox"]
            colour = COLOURS[(a["category_id"] - 1) % len(COLOURS)]
            draw.rectangle([x - 3, y - 3, x + w + 3, y + h + 3], outline=colour, width=2)
            labels.append(SHORT.get(cats[a["category_id"]], cats[a["category_id"]]))
        caption = (
            f"{im['capture_id'][:8]} · board {im['board']} {im['pose']} · d4={im['d4']}"
            + (f" · CT→{im['ct_target'][:8]}" if im["ct_target"] else "")
            + f"<br>{', '.join(labels) or 'negative'}"
        )
        cards.append(
            f'<figure><img src="data:image/jpeg;base64,{_b64(tile.resize((320, 320)))}">'
            f"<figcaption>{caption}</figcaption></figure>"
        )
    html = (
        "<!doctype html><meta charset='utf-8'><title>QA tiles</title>"
        "<style>body{font:13px system-ui;margin:16px;background:#f4f4f2}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}"
        "figure{margin:0}img{width:100%;border-radius:4px}</style>"
        f"<h1>{ds_id} — dev_all_train sample (box colours: NAL red, Brücke blue, Fahne yellow)</h1>"
        f"<main>{''.join(cards)}</main>"
    )
    out = ds_dir / "qa.html"
    out.write_text(html, encoding="utf-8")
    logger.info(f"QA gallery: {out} ({len(cards)} tiles)")
    return out
