# zander-ml — solder-defect detector toolchain

Trains and benchmarks the solder-joint defect detector on the photos collected
by the zander pipeline. **Required output per photo (user, 2026-09-28): N.iO +
which defect class(es) were detected, or i.O.** Classes: *Nicht ausreichend
Lot* (NAL), *Brücke*, *Fahne*. Research, model plan and evaluation rationale:
`../../reports/PCB solder defect detection plan.md`.

- **Two class schemas from one pipeline** (`dataprep/remap.py`): `def3` (all
  three classes — the product requirement) and `nal1` (NAL only — the
  comparison: the 3-class model must not be worse on NAL).
- **640-px tiling everywhere** — defects have a ~28–38 px median side at the
  native 2616×1960, so photos are sliced with the pinned grid
  (`infer/tiling.py`), trained on tiles, and evaluated stitched at photo level
  (per-class IoS merge across tile seams).
- **Frozen, board-aware split** (`datasets/split_v2.json`): whole board groups
  only; `test` is the one-shot honest set — **never** used for selection or
  threshold tuning.
- **One decision rule** (`infer/verdict.py`) shared by evaluation and the
  inspector: N.iO if any detection ≥ t, classes = those above t.

## Setup

```bash
cd ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt     # local (CPU torch pins + everything)
# Colab installs requirements.txt and uses its preinstalled CUDA torch.
```

## The flow

```bash
cd ml                                   # every command runs from here

python cli.py attributes                # audit -> committed photo_attributes_v1.csv
python cli.py ingest                    # photos + Aug-17 labels + attributes -> manifest (+gates)
python cli.py split                     # board-aware groups + frozen split_v2
python cli.py tile --schema def3 --zip  # tiles + offline augmentation + COCO + YOLO view + Colab zip
python cli.py tile --schema nal1 --zip
python cli.py qa --schema def3          # eyeball augmented training tiles (boxes on the defects?)
python cli.py report --schema def3      # committed dataset card -> reports/dataset_card.md

# training (Colab: notebooks/colab_train.ipynb; local smoke: --device cpu):
python cli.py train --config configs/matrix/yolo26n_def3_fold0.yaml

# bring a Colab bundle home and register it:
python cli.py import-run ../ml_data/run_dirs/yolo26n_def3_fold0_bundle.zip
mlflow ui --backend-store-uri file:../ml_data/mlruns

# photo-level evaluation (each fold model on ITS fold), then freeze the threshold:
python cli.py eval --model <bundle>/model.onnx --schema def3 --bucket fold0 --run-name y26n_def3_f0
python cli.py choose-op y26n_def3_f0 y26n_def3_f1 y26n_def3_f2       # -> operating_point_*.json
python cli.py compare y26n_def3_f0 y11s_def3_f0                       # paired, board-cluster bootstrap
# final model only, once: --operating-point is mandatory for test / probe_xprod / future
python cli.py eval --model final.onnx --schema def3 --bucket test --operating-point <op.json>
```

## Data (snapshot v2, 2026-09-28)

- **Photos**: `data_fotos/Bilder 27-07-26/` (481, Jun–Jul, Pi era) +
  `data_fotos/labeled 2/` (85, Aug; unlabeled-format captures despite the name).
  **Labels**: the Aug-17 Label Studio YOLO export (superset of July-27 v1).
- **Photo attributes** (`datasets/photo_attributes_v1.csv`, anonymised, from the
  2026-09-28 visual audit): content (PCB / no PCB / placeholder …), board
  (A / A2 / B), pose (seated / handheld), lighting. Ingest refuses photos
  without a row — new sessions need a quick audit first.
- **Quarantine**: 87 photos without a usable PCB (audit `content`), incl. the two
  empty-fixture frames that carried boxes. **Review** (excluded until fixed in
  LS): iO-with-box and "Weitere Überprüfung"-without-box.
- **Unannotated photos are never negatives** (235, of which ~150 usable PCB photos).
- **Known confounds** (see the dataset card): NAL only on board A/A2;
  Brücke/Fahne ≈ board B = **one panel, one session**; 170 of 194 iO photos are
  handheld (not the production pose).

## Split v2 (`dataprep/grouping.py`, `dataprep/split.py`)

Groups = time bursts (≤ 30 s) **cut at evidenced board changes**, plus re-shot
links (same defect set at a non-systematic position — this is how the retired
split_v1 leaked: re-shots of one board sat in fold0/fold2/test), plus board B
as one group. Buckets: `test` (one-shot; ~20 % of NAL photos per session + a
share of the scarce seated iO photos), `fold0..2`, `train_always` (board B — the
only Brücke/Fahne source, so it is in every training view and never validated),
`future` (rule: captures after the freeze). **Probe**: the 16 board-A/A2
Brücke/Fahne boxes are held out of every training view — the only honest
cross-product check (`--bucket probe_xprod`, hit counts, not rates).

## Augmentation policy (principle: only against measured nuisance variation, symmetric for positives and negatives)

| | where | why |
|---|---|---|
| D4 rot90/flip per tile | offline (`dataprep/tiler.py`) | orientation-free defects; exact on boxes; same orientation mix for positives and negatives |
| class-aware repeat-factor sampling | offline | Brücke tiles outnumber NAL; target ~30 % positive entries |
| cross-board colour transfer (p = 0.3) | offline | attacks the colour part of the class ↔ board shortcut |
| handheld-negative cap, partial-box drop, probe drop | offline | production-pose balance; no "defect = background" slivers; honest probe |
| flips, scale ≤ 0.2, HSV (v 0.35, s 0.4, h 0.015), mosaic + close_mosaic | online (`train/ultralytics_run.py`) | session exposure spread 121–177 luma; fixed camera distance |
| **off**: free rotation, mixup, noise/JPEG, copy-paste, diffusion | – | inflates tiny boxes / blends defects / not measured / weak evidence |

Ablations (`cli.py tile` flags `--no-d4`, `--ct-p`, `--no-handheld-cap`,
`--fixed-k`) get their own dataset ids (`ds_v2_def3_nod4`, …).

## Evaluation (`eval/photo_scorer.py`, `eval/run_eval.py`)

1. **Verdict** (acceptance): photo AUROC; escape rate; false-call rate on
   *seated* iO photos (handheld separately); Wilson CIs; per board × pose.
2. **Which error**: per-class photo sensitivity and class false alarm (incl. on
   iO photos per board = background-shortcut test), exact class-set accuracy,
   class confusion. A wrong class is not an escape.
3. **Defect** (model selection): per-class and class-agnostic recall @ 0.5 FPPI
   (IoU 0.3) with **board-group** cluster-bootstrap CIs; box-level confusion; AP50
   for literature comparison only.

Operating point: highest t with pooled out-of-fold photo sensitivity ≥ 0.95
(`choose-op`), frozen into `operating_point_*.json`; the inspector's
`CONFIDENCE_THRESHOLD` must be set to it. Selection: rank by NAL recall @ 0.5
FPPI across folds; a winner must beat the runner-up with a paired CI excluding 0
(`compare`), otherwise the faster model wins. Results: `reports/results_v2.csv`
(one row per eval; rows are only comparable with equal split_id, snapshot_id,
tiler_version and dataset fingerprint).

## Candidates

YOLO26n / YOLO26s / YOLO11s (Ultralytics, AGPL-3.0) — `configs/matrix/*_def3_fold*.yaml`,
plus `yolo26n_nal1_fold*` as the NAL-only comparison. D-FINE-S trained with
**DEIM v1** (Apache-2.0, pinned `09d35d5`; `train/deim_run.py --schema def3`) —
**not DEIMv2** (non-commercial licence since 2026-08-24).

## History

`split_v1.json` / `groups.json` / `results.csv` belong to the retired v1
(481 photos, NAL only, time-burst groups that leaked re-shots). They stay
committed for traceability; nothing reads them.
