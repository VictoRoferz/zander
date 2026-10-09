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

## Runbook: Mac → Drive → Colab → Mac → inspector

Every command runs from `ml/` (`cd ml`); `python cli.py <cmd> --help` has all
options. `<F>` is a **family** = `<arch>_<schema>` (e.g. `yolo26n_def3`); its
runs are `<F>_fold0..2` (one model per validation fold) and `<F>_devall` (the
final retrain on all dev folds). Eval runs are named `<F>_fold<k>` on a model's
own fold and `<bundle_run>__<bucket>` otherwise (`docs/CONTRACTS.md` §8).

**1 · Data (Mac, once per label snapshot)**

```bash
python cli.py attributes                # audit -> committed photo_attributes_v1.csv
python cli.py ingest                    # photos + Aug-17 labels + attributes -> manifest (+gates)
python cli.py split                     # board-aware groups + frozen split_v2 (refuses to overwrite)
python cli.py tile --schema def3 --zip  # tiles + offline augmentation + COCO + YOLO view + Colab zip
python cli.py tile --schema nal1 --zip
python cli.py qa --schema def3          # eyeball augmented training tiles (boxes on the defects?)
python cli.py report --schema def3      # committed dataset card -> reports/dataset_card.md
```

**2 · Drive.** Upload the two tile zips to `MyDrive/ZanderML/` as
`ds_v2_def3_tiles.zip` / `ds_v2_nal1_tiles.zip` (company photos — supervisor OK
first). Drive layout: `docs/CONTRACTS.md` §8.5.

**3 · Colab (GPU) — the training matrix.** Open `notebooks/colab_train.ipynb`
from the pushed branch and follow **`notebooks/README.md`** (GPU choice, matrix
cell, resume after a disconnect, timings). Every finished run lands as
`MyDrive/ZanderML/bundles/<run>_bundle.zip` (+ `.sha256`, the "finished"
marker). A local CPU smoke run uses the same entry point:
`python cli.py train --config configs/smoke_def3.yaml --device cpu`
(`--set KEY=VALUE` overrides, `--mirror-dir` for resume).

**4 · Mac — score every fold model on its own fold**

```bash
# bundles downloaded to ml_data/run_dirs/ (or point --bundles at the Drive folder)
python cli.py eval-matrix --bundles "../ml_data/run_dirs/*_bundle.zip"
mlflow ui --backend-store-uri file:../ml_data/mlruns   # experiment "zander-v2"
```

Per bundle: import into `ml_data/bundles/<run>/` + MLflow (verifies the
`.sha256` marker; a different bundle under an existing run name needs
`import-run --force`, which archives the old one), evaluate `<F>_fold<k>` with
the adapter/schema from `meta.json`, log the headline numbers into the bundle's
MLflow run (`log-eval`). Finished runs are skipped on a rerun; dev_all bundles
are skipped (no out-of-fold bucket); a failing bundle is reported and the
matrix goes on. Galleries are off (`--gallery` to turn on).

**5 · Freeze one operating point per family**

```bash
python cli.py choose-op --family yolo26n_def3   # -> ml_data/eval/operating_point_def3_yolo26n_def3.json (+ .md)
```

t* = the highest threshold with pooled out-of-fold photo sensitivity ≥ 0.95.
Refused unless the three runs are three **distinct** fold models, each scored
on its own fold, trained under the run policy (last epoch, no early stopping,
every epoch completed), with equivalent training configs (arch, dataset_id, epochs,
imgsz, batch actually used, augment, final_weights, training `extra` kwargs —
validation/logging/runtime keys such as `val`, `plots`, `workers` excluded —,
dataset fingerprint; seeds may differ); smoke runs need
`--allow-smoke`. The file is frozen: `--force` archives it to `ml_data/eval/archive/`.

**6 · Rank and gate (paired decisions, never overlapping CIs)**

```bash
python cli.py summary                                            # -> reports/model_selection_v2.md + ml_data/eval/summary_v2.json
python cli.py compare yolo26n_def3 yolo11s_def3 --class NAL                  # G1: top-1 vs top-2
python cli.py compare yolo26n_def3 yolo26n_nal1 --class NAL --margin 0.05    # NAL gate
```

`summary` and `compare` pool a family's fold runs only under choose-op's guards
(distinct own-fold models, one recipe). `summary` scores them at the family's
frozen operating point (self-tuned and flagged ⚠ if `choose-op` has not run),
ranks the def3 families by pooled NAL recall @ 0.5 FPPI, and runs G1 and the
NAL gate itself. **G1:** a winner needs a paired CI excluding 0, otherwise the
faster/simpler model wins. **NAL gate:** def3 must be non-inferior to nal1 on
NAL recall within 0.05 (CI lower bound of def3 − nal1 > −0.05). `compare`
takes eval runs or families (pooled over `<F>_fold0..2`); `--class` accepts a
name, short name (NAL/Brücke/Fahne), run A's class index, or `any`; escapes use
each side's frozen operating point (`n/a` without one).

**7 · Colab — retrain the winner** on all dev folds with its
`configs/final/<F>_devall.yaml` (notebook FINAL cell) → `<F>_devall_bundle.zip`.

**8 · Mac — probe and the one-shot test**

```bash
python cli.py import-run ../ml_data/run_dirs/yolo26n_def3_devall_bundle.zip
OP=../ml_data/eval/operating_point_def3_yolo26n_def3.json
python cli.py eval --model ../ml_data/bundles/yolo26n_def3_devall/model.onnx --bucket probe_xprod --operating-point $OP
python cli.py eval --model ../ml_data/bundles/yolo26n_def3_devall/model.onnx --bucket test --operating-point $OP
git add reports/test_ledger.csv       # the look is recorded — commit it
```

**9 · Export** (the only way a model reaches production)

```bash
python cli.py export --bundle ../ml_data/bundles/yolo26n_def3_devall --operating-point $OP
# -> ml_data/exports/detector.onnx + detector.json (docs/CONTRACTS.md §1)
```

Checks before writing: bundle meta vs operating point (schema, classes, split,
tiler, and the test eval's identity check: family, the op's fold bundles still
hold the models it was frozen from, config equivalence with them); the bundle
is the dev_all retrain (`--allow-fold-model` records an exception); a test eval
of exactly this model at exactly this operating point exists — the latest such
look, `__test__look<k>` included (`--no-test-eval` records null, but cannot waive
a test look the model already had at another operating point); **parity** —
the deployed pipeline (`conf_floor = t`) reproduces the evaluated verdicts on
the 5 photos nearest the threshold, zero mismatches.
`--force` archives an existing export to `exports/archive/<timestamp>/`.

**10 · Inspector.** Copy `detector.onnx` **and** `detector.json` to
`DATA_ROOT/models/` (Windows: `deploy\data\models\`; the container reads
`MODEL_PATH=/data/models/detector.onnx`) and restart the inspector;
`GET :8004/api/v1/status` shows the card. The card is authoritative:
threshold, adapter and class names come from it, not from the environment.

**Crop verifier (optional second stage, offline prototype):** `python cli.py
verifier ...` — see **`verifier/README.md`**.

### Honesty guards (enforced in code, `eval/run_eval.py`)

- `eval` refuses, before any photo is read: an unknown bucket; a schema,
  adapter or split_id that differs from the bundle's `meta.json` (never
  overridable); a fold bucket scored by a model that trained on it (another
  fold's model, or a dev_all model); a model without `meta.json`; a non-dev_all
  model on `test`; a bundle that breaks the run policy (`weights: best`,
  `extra.patience` ≠ 0, `epochs_completed` < `epochs`: a checkpoint selected on
  or stopped by its validation view). `--allow-foreign-fold` overrides the last
  four for diagnostics only — the run is stamped `guard_overridden` and never
  recorded; `choose-op` never pools such a model. A `model.onnx` that is not
  the file its `meta.json` (`onnx.sha256`) / import marker describes is
  refused outright.
- `choose-op`, `compare` and `summary` judge an eval run by the model it
  scored: if the bundle now holds another model (re-import `--force`) the run is
  a **stale eval** (re-run it); a frozen operating point is applied only to the
  fold models it was frozen from (else stale: re-run `choose-op --force`).
- `test`, `probe_xprod` and `future` need `--operating-point` with a numeric
  threshold in (0, 1] (`--conf` may not exceed it); `probe_xprod` never
  includes test photos. On `test` the operating point must come from
  `choose-op` (family, sources, target sensitivity) and belong to the model —
  same family, training config equivalent to its fold bundles — checked before
  the one-shot look is spent.
- **Test ledger** (`reports/test_ledger.csv`, committed): every look at `test`,
  keyed by model, operating point, `--conf` floor and the ground truth scored
  (snapshot id + `gt_sha256`). Re-running the same key is fine; anything else —
  another model, op, floor, or a relabel — is a second look and needs
  `--second-look "<reason>"` (recorded in the ledger, `metrics.json` and
  `report.md`). Test evals in `ml_data/eval` (shared by every checkout and
  worktree) count as looks even when this checkout's ledger lost them. A further
  look lands in `<bundle_run>__test__look<k>`; a look's eval dir is never
  overwritten.
- `reports/results_v2.csv` gets one row per honest eval (never smoke or
  overridden runs, never `train_always`; a `probe_xprod` row keeps only the
  probe hit counts — the photo-level columns stay empty). Smoke rows are never
  committed — `git checkout -- reports/results_v2.csv` after experiments.
- `compare` refuses two sides scored against different ground truth (a
  relabel between the evals). `choose-op` refuses a target sensitivity outside
  (0, 1] and a t* below a fold run's `--conf` floor; the op JSON records
  `allow_smoke`.

### Commands

| command | does |
|---|---|
| `train` | one run config → bundle zip (`train/dispatch.py`; Colab or `--device cpu`) |
| `import-run` | bundle zip/dir → `ml_data/bundles/<run>/` + MLflow run (idempotent; `--force`) |
| `eval-matrix` | import + score every fold bundle on its own fold + log to MLflow |
| `eval` | one model on one bucket → `ml_data/eval/<run>/{metrics.json, predictions.json, report.md, gallery.html}` |
| `log-eval` | an eval run's headline numbers → its bundle's MLflow run |
| `choose-op` | a family's pooled OOF runs → frozen `operating_point_<schema>_<family>.json` |
| `compare` | paired comparison of two runs or families (`--class`, `--margin`) |
| `summary` | pooled OOF table, ranking, G1, NAL gate → `reports/model_selection_v2.md` |
| `export` | verified `detector.onnx` + `detector.json` for the inspector |
| `latency` | tiled-inference latency bench on this host |
| `verifier` | crop-verifier prototype (arguments go to `verifier/cli.py`) |

Eval artefacts: `predictions.json` v2 adds `format_version`, `source{…}`
(model sha256, bundle run, fold, adapter, conf floor) and per-photo
`width`/`height`, and stores scores at full precision — a threshold frozen
from them is exactly reproducible by the deployed detector. v1 files still load.

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
(`choose-op`), frozen into `operating_point_<schema>_<family>.json`; `export`
writes it into `detector.json`, which the inspector reads (no manual
`CONFIDENCE_THRESHOLD`). Selection: rank by pooled NAL recall @ 0.5 FPPI
(`summary`); a winner must beat the runner-up with a paired CI excluding 0
(`compare`), otherwise the faster model wins. Results: `reports/results_v2.csv`
(one row per eval; rows are only comparable with equal split_id, snapshot_id,
tiler_version and dataset fingerprint) and `reports/model_selection_v2.md`.

## Candidates

YOLO26n / YOLO26s / YOLO11s (Ultralytics, AGPL-3.0) — `configs/matrix/*_def3_fold*.yaml`,
plus `yolo26n_nal1_fold*` as the NAL-only comparison. D-FINE-S trained with
**DEIM v1** (Apache-2.0, pinned `09d35d5`) — `configs/matrix/deim_s_def3_fold*.yaml`,
run like any config (`python train/dispatch.py --config configs/matrix/deim_s_def3_fold0.yaml`,
or the notebook's optional `DEIM_MATRIX`; plan with `python train/deim_run.py --config … --dry-run`) —
**not DEIMv2** (non-commercial licence since 2026-08-24). DEIM-S measured ≈60 ms per tile
(≈1.2–1.35 s per photo) on the Mac CPU, over the 1 s budget: if it wins on accuracy,
DEIM-N is the next candidate.

## History

`split_v1.json` / `groups.json` / `results.csv` belong to the retired v1
(481 photos, NAL only, time-burst groups that leaked re-shots). They stay
committed for traceability; nothing reads them.
