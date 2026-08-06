"""
Batch inspection CLI — run the loaded model over a folder of photos without
the HTTP service:

    python batch.py <folder-of-jpgs> [--out results.jsonl]

One JSON line per photo: {"file", "count", "detections", "latency_ms"}.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from config.settings import settings
from services.detector_service import detector_service

logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("inspector.batch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results.jsonl"))
    args = parser.parse_args()

    detector_service.load()
    if not detector_service.ready:
        LOG.error(f"no model at {settings.model_path}")
        return 1

    photos = sorted(args.folder.glob("*.jpg"))
    if not photos:
        LOG.error(f"no .jpg files in {args.folder}")
        return 1

    with open(args.out, "w", encoding="utf-8") as f:
        for n, path in enumerate(photos, 1):
            result = detector_service.inspect_bytes(path.read_bytes())
            f.write(json.dumps({"file": path.name, **result}) + "\n")
            LOG.info(
                f"[{n}/{len(photos)}] {path.name}: {result['count']} detection(s) "
                f"in {result['latency_ms']} ms"
            )
    LOG.info(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
