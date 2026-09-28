"""
Batch inspection CLI: run the loaded model over a folder of photos without
the HTTP service:

    python batch.py <folder-of-jpgs> [--out results.jsonl]

One JSON line per photo: {"file", "verdict", "count", "detections", ...}
(the /inspect response fields). Undecodable photos are logged and skipped.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from services.detector_service import ImageDecodeError, detector_service

logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("inspector.batch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results.jsonl"))
    args = parser.parse_args()

    detector_service.load()
    if not detector_service.ready:
        LOG.error(f"no model loaded: {detector_service.load_error}")
        return 1

    photos = sorted(args.folder.glob("*.jpg"))
    if not photos:
        LOG.error(f"no .jpg files in {args.folder}")
        return 1

    skipped = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for n, path in enumerate(photos, 1):
            try:
                result = detector_service.inspect_bytes(path.read_bytes())
            except ImageDecodeError as e:
                skipped += 1
                LOG.warning(f"[{n}/{len(photos)}] {path.name}: skipped ({e})")
                continue
            f.write(json.dumps({"file": path.name, **result}, ensure_ascii=False) + "\n")
            LOG.info(
                f"[{n}/{len(photos)}] {path.name}: {result['verdict']}, "
                f"{result['count']} detection(s) in {result['latency_ms']} ms"
            )
    LOG.info(f"written: {args.out} ({len(photos) - skipped} photos, {skipped} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
