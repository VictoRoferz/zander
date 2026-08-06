# zander-ml — NAL defect detector toolchain

Trains and benchmarks the **"Nicht ausreichend Lot"** (insufficient solder)
detector on the photos collected by the zander pipeline. Design decisions,
dataset analysis and the milestone plan live in the project plan (2026-08);
the short version:

- **Single class** (Brücke/Fahne stay labeled in Label Studio but are dropped
  to background at remap; recorded as `other_boxes` for FP diagnosis).
- **640-px tiling everywhere** — defects have a ~28 px median side at the
  native 2616×1960, so photos are sliced with the pinned TILER v1 grid
  (`infer/tiling.py`), trained on tiles, and evaluated stitched at photo level
  (IoS non-max merge across tile seams).
- **Frozen, group-aware split** (`datasets/split_v1.json`): whole time-burst
  groups only, session-stratified; `test` is the one-shot honest set —
  **never** used for selection or threshold tuning.
- **Candidates** (round 1): YOLO26n / YOLO26s / YOLO11s (Ultralytics) +
  DEIM/D-FINE-S (Apache, arm's-length via `train/deim_run.py`). Every model
  is scored by the SAME stitched evaluator from raw ONNX — no per-model
  post-processing tuning.
- **Operating metric**: recall @ ≤0.5 false calls per photo (FROC, IoU 0.3
  matching), with bootstrap CIs. AP50 / AP50-95 are reported for literature
  comparability only.

## Setup

```bash
cd ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt     # local (CPU torch pins + everything)
# Colab installs requirements.txt and uses its preinstalled CUDA torch.
```

## The flow

```bash
cd ml                       # every command runs from here (service-style)

python cli.py ingest        # photos + label snapshot -> datasets/manifest.json (hard gates)
python cli.py split         # time-burst groups + frozen grouped split (refuses to overwrite)
python cli.py tile --zip    # 640px tiles + COCO GT + YOLO view + Colab zip (~1 GB)
python cli.py report        # committed dataset card -> reports/dataset_card.md

# latency sizing on this machine (any raw ONNX):
python cli.py latency --model ../ml_data/pretrained/yolo11n.onnx

# training (Colab: notebooks/colab_train.ipynb; local smoke: --device cpu):
python cli.py train --config configs/matrix/yolo26n_fold0.yaml

# bring a Colab bundle home and register it:
python cli.py import-run ../ml_data/run_dirs/yolo26n_fold0_bundle.zip
mlflow ui --backend-store-uri file:../ml_data/mlruns   # browse runs

# the number that decides the winner (stitched photo-level eval):
python cli.py eval --model <bundle>/model.onnx --adapter ultralytics --bucket fold0
# -> appends one row to reports/results.csv + writes report/gallery under ml_data/eval/
```

## Benchmark protocol (M2/M3)

1. Train each `configs/matrix/*.yaml` on Colab (9 Ultralytics runs; ~10 GPU-min
   each) + DEIM-S per fold (`train/deim_run.py`, pin `DEIM_COMMIT` first).
2. Import every bundle, then `eval` each on ITS OWN fold (`--bucket foldN`).
3. Compare candidates on mean ± CI of `recall_at_0.5fppi` across folds in
   `reports/results.csv`. Ties → lower latency wins (`cli.py latency`).
4. Winner: retrain on `dev_all`, evaluate ONCE with `--bucket test`, write the
   model card, export to `ml_data/exports/nal_detector.onnx`, set the
   inspector's `CONFIDENCE_THRESHOLD` to the dev-fold t*.

Rules that keep the numbers honest: nothing from `test` influences any
decision; per-tile/in-training mAP never enters the comparison; every eval row
records (split_id, snapshot_id, tiler_version, dataset_fingerprint) — if any
of those change, old rows are not comparable.

## Data notes

- Canonical inputs: `data_fotos/Bilder 27-07-26/` (481 photos + sidecars) and
  the July-27 LS YOLO export as label **snapshot v1**. The YOLO export is
  lossy (drops iO/NiO choices, rotation) — before the final test eval, pull a
  fresh **full-JSON** export from the live Label Studio (Windows PC) as
  snapshot v2 and re-run `ingest` (gates will flag the re-baseline).
- 183 photos are still unannotated: they are excluded (NOT negatives) and
  pre-assigned to split buckets so labeling them later changes nothing else.
- Quarantined junk (12 dark/webcam frames) and the review-list image
  (`0fef68e4…`, iO-with-Fahne-box contradiction) are listed in the manifest.

## Follow-ups (designed, not wired)

- **Inspector service** (`../inspector/`, :8004) serves the winner ONNX via
  the same `ml/infer` pipeline (torch-free). Receiver hook (BackgroundTasks
  after ingest → `unlabeled/<id>.pred.json` sidecar), dashboard badge/overlay,
  and launcher/compose entries are documented in `../inspector/main.py`.
- **Pre-annotation flywheel**: serve `best.pt` through Label Studio's YOLO ML
  backend so annotators correct instead of draw — the fastest route to
  labeling the remaining 183 photos and future captures.
- **Data growth priorities**: seeded-defect boards for Brücke (6 boxes today
  → target ≥100 before it becomes a class again), keep capturing under the
  July-20 setup.
