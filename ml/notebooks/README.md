# Colab training runbook (zander-ml, split_v2)

`colab_train.ipynb` trains the out-of-fold matrix on a Colab GPU, and the final dev_all model after it.
Every run ends as a **bundle**, `<run>_bundle.zip` plus its `.sha256`, on Google Drive. You evaluate the bundles on the Mac afterwards; see `ml/README.md`.

What the notebook runs is `train/matrix.py`. That driver:
- skips finished runs;
- resumes a partial run from its Drive mirror;
- retries a failed run once, and never lets one failure stop the loop;
- logs every attempt to `matrix_log.jsonl`.

## 0. Before the first session

1. **Data governance.** Confirm with your supervisor that the Zander production photos may be stored on Google Drive. The tile zips contain them.
2. **Colab plan.** You need a plan with compute units (Colab Pro) for an **L4** GPU.
   - An L4 costs about **5 compute units per hour**. Check the current rate in Colab.
   - The full 12-run matrix is about **11–17 h**, which is **55–85 CU**, spread over **at least 2 sessions**.
3. **The code must be on GitHub.** The notebook clones `BRANCH` from `https://github.com/VictoRoferz/zander`. Code is pushed only after review, so make sure the branch you train from has been pushed.

## 1. Build the tile zips (Mac)

```bash
cd ml && source .venv/bin/activate
python cli.py tile --schema def3 --zip   # -> ../ml_data/ds_v2_def3_tiles.zip  (about 1.3 GB)
python cli.py tile --schema nal1 --zip   # -> ../ml_data/ds_v2_nal1_tiles.zip  (about 1.5 GB)
```

Each zip carries the tiles, the COCO ground truth and `provenance.json` with the dataset fingerprint.
Colab rebuilds the YOLO view from the **committed** manifest and split, then asserts that the fingerprints match. So build the zips from the same commit you push.
You need to rebuild them only when the manifest, the split or the tiler changes. The notebook's fingerprint assert tells you when that happened.

## 2. Upload to Google Drive

1. Create the folder `MyDrive/ZanderML/`.
2. Upload both zips there without renaming them: `ds_v2_def3_tiles.zip` and `ds_v2_nal1_tiles.zip`. Use drive.google.com or Google Drive for desktop.

The notebook creates `bundles/`, `mirror/`, `logs/` and `matrix_log.jsonl` next to them.

## 3. Open the notebook and pick the runtime

1. Open the notebook straight from the branch:
   `https://colab.research.google.com/github/VictoRoferz/zander/blob/feature/ds-v2-preprocessing/ml/notebooks/colab_train.ipynb`.
   If that URL does not resolve, use Colab → File → Open notebook → GitHub, enter `VictoRoferz/zander`, choose the branch and pick `ml/notebooks/colab_train.ipynb`.
   After the merge into `main`, use `.../blob/main/...` and set `BRANCH = 'main'` in the Parameters cell.
2. Choose **Runtime → Change runtime type → L4 GPU** and turn **High-RAM** on.
3. Optional: File → Save a copy in Drive, so that your parameter edits persist.

## 4. First session: time one run

1. In the **Parameters** cell, add `MATRIX = MATRIX[:1]` below the list. That trains only `yolo26n_def3_fold0`.
2. Choose **Runtime → Run all** and accept the Drive permission prompt.
3. The first cells check the GPU, mount Drive, clone the branch (printing the git sha), and assert that `ml/datasets/split_v2.json` exists, so that v1 code is refused.
4. They then install `requirements-colab.txt`, unpack the zips and rebuild the views.
5. Setup takes about 5 min. Then watch the epoch time in the output.
   - On an L4 a 120-epoch fold run should take about **50–90 min**: n-scale at the low end, s-scale at the high end.
   - nal1 runs take about **35 % longer**, because their views are larger.
6. If one epoch takes much longer than about 45 s for yolo26n, the dataloader is starved. Check the vCPU count that the GPU cell prints, and see Troubleshooting.

## 5. The matrix

1. Remove the `MATRIX[:1]` line.
2. **Run all.** The finished run is skipped, and the others train in priority order:
   1. `yolo26n_def3` fold 0–2;
   2. `yolo26n_nal1` fold 0–2, the NAL-only comparison;
   3. `yolo26s_def3`;
   4. `yolo11s_def3`.

   Each block of three is a complete family for `choose-op`. If the budget runs short, stop after a block.
3. After every run the driver prints an ETA from the measured minutes per epoch. At the end it prints a summary table.
4. Keep the browser tab open. Colab disconnects idle sessions, and background execution needs Pro+.
5. The **last cell** runs `drive.flush_and_unmount()`. Google Drive uploads in the background, so this waits until everything the session wrote (bundles, mirror, logs) is really on Drive. If you stop a session early, run that cell by hand before you disconnect.

A run's `mirror/<run>/` is **not** deleted when its bundle is written: Drive may still be uploading the bundle. The next session that reads the bundle back from Drive and verifies its sha deletes the mirror.

**After a disconnect:** choose **Reconnect → Runtime → Run all**.
- Runs whose bundle and `.sha256` are on Drive are skipped.
- A partial run resumes from `mirror/<run>/`. The mirror is refreshed at most every 5 min, plus at the last epoch, so at most about 5 min of training is lost.
- Each resume shows up in the bundle's `meta.json` as another entry in `attempts`.

**Monitoring:**
- the notebook output;
- `logs/<run>.log` on Drive, which holds the full output of every attempt;
- the last cell, which gives the latest status per run across sessions. `started` means the session died mid-run and will resume.

## 6. Final retrain (after the Mac-side evaluation)

1. Evaluate the fold bundles on the Mac and let `choose-op` / `compare` pick a family (see `ml/README.md`).
2. Set `FINAL = ['configs/final/<family>_devall.yaml']`, for example `yolo26n_def3_devall`.
3. **Run all.** The matrix part skips everything that is already done, and the final cell trains the dev_all model. That takes about 30–40 % longer than a fold run, because it uses all dev photos.
4. The final model uses its family's fold-run operating point. It never gets its own.
5. `final_weights: last` is mandatory here. The runner refuses `best`, because the "val" view of dev_all lies inside its training data.

## 7. Bring the bundles home

```bash
# download MyDrive/ZanderML/bundles/*_bundle.zip together with the .sha256 files into ml_data/run_dirs/, then:
cd ml_data/run_dirs && shasum -a 256 -c yolo26n_def3_fold0_bundle.zip.sha256
```

Then import and evaluate, following `ml/README.md`: `import-run` and `eval` (each fold model on its own fold), then `choose-op` and `compare`.

Each bundle holds:
- `config.yaml`, the **effective** config, whose sha256 is in `meta.json`;
- `model.onnx`, plus `model_raw.onnx` for YOLO26;
- `last.pt`;
- `metrics.json`, `args.yaml`, `pip-freeze.txt` and `meta.json`.

The format is described in `docs/CONTRACTS.md` §8.1.

## 8. Optional: DEIM / D-FINE-S, the Apache-2.0 challenger

The Ultralytics models are AGPL-3.0. D-FINE-S trained with **DEIM v1** is the license-clean alternative: the DEIM repo is pinned to one commit, its Apache-2.0 LICENSE is checked (text and sha256) at every run, and each bundle ships it as `LICENSE_DEIM`. DEIMv2 and EdgeCrafter are **not** used: they are non-commercial since 2026-08-24.

**GPU budget.** This is an estimate, because no DEIM run has been timed yet:
- about **1.5–3.5 h per 80-epoch fold run** on an L4. DETR-family training costs roughly 1.5–3× an s-scale YOLO epoch;
- about **5–10 h for the three folds**, which is **about 25–50 compute units**;
- the dev_all retrain takes about 40 % longer than a fold run, because it has 4.6k instead of 3.0–3.5k tiles.

**Run it:**
1. In the Parameters cell, set `DEIM_MATRIX` to the three configs from the comment above it (`configs/matrix/deim_s_def3_fold{0,1,2}.yaml`). The first time, add `DEIM_MATRIX = DEIM_MATRIX[:1]` to time one run.
2. **Run all.** The DEIM cell runs after the YOLO matrix:
   - it installs DEIM's extra deps (`python train/deim_run.py --install-deps`: pinned faster-coco-eval, calflops and onnxruntime). Colab's torch is not touched;
   - it puts the init checkpoint in place (`--fetch-weights`): DEIM's COCO-pretrained DEIM-S from the model zoo of the pinned README, which is hosted on Google Drive;
   - it then trains through `train/matrix.py`, like the YOLO runs: skip, resume and retry work the same way.
3. If Google Drive refuses the download (quota), download the "S" checkpoint of DEIM's model zoo by hand and upload it as `MyDrive/ZanderML/pretrained/deim_hgnetv2_s_coco.pth`. The cell copies it from there in every later session too. Its sha256 is pinned on first use, in `<file>.sha256` next to it.
4. If DEIM wins on the Mac: `FINAL = ['configs/final/deim_s_def3_devall.yaml']`. The DEIM cell then installs the deps before the final cell runs.

**How the DEIM runs differ from the YOLO runs** (details: the docstring of `train/deim_run.py`):
- 80 epochs at batch 16, fine-tuned from the COCO checkpoint. Batch 32, DEIM's COCO batch, would not fit an L4 at the 800-px multi-scale maximum.
- The schedule follows from the epochs and the view size: 8 epochs without photometric aug at the end, LR warm-up over 2 epochs, EMA memory of about 2.7 epochs.
- The **last** epoch is bundled. DEIM's own "stage 2", which reloads the checkpoint that scored best on the fold's validation photos, is switched off. That would be a selection leak into the out-of-fold evaluation.
- The augmentation mirrors the YOLO policy: flips plus ColorJitter at the Ultralytics HSV strengths. Mosaic, zoom-out and IoU crops are off, because they shrink or rotate 28-px defects.
- The mirror holds `last.pt` (DEIM's full checkpoint), `log.txt` and `zander_run.json`. The bundle also holds `LICENSE_DEIM` and `deim_override.yml`; its `last.pt` is the EMA checkpoint the ONNX was exported from.

## Troubleshooting

| symptom | meaning / fix |
|---|---|
| `has no ml/datasets/split_v2.json` | `BRANCH` points at v1 code; use the v2 branch |
| `fingerprint mismatch` | The zip and the branch were built from different data or code. Rebuild the zips from the pushed commit |
| `missing .../ds_v2_*_tiles.zip` | Upload the zip to `MyDrive/ZanderML/` without renaming it |
| `WARNING ... config_sha256 differs` | A finished bundle was trained with a different effective config (an edited config or `EXTRA_SET`). It is **kept, never overwritten**. Delete the bundle and its `.sha256` on Drive to retrain |
| a run `FAILED` twice | Read the error in the summary or in `logs/<run>.log`, then Run all again. A partial run resumes |
| `config/data error (not retried)` | Exit code 2 is a configuration or dataset problem. Fix it first, because retrying won't help |
| CUDA out of memory | Should not happen on an L4 at batch 16. Do not lower `batch`: it is part of the compared config. Report the problem, or use an A100 |
| very slow epochs, few vCPUs | With ≤ 4 vCPUs the GPU profile turns on `cache=disk`, so the first epoch is slower while it caches. Prefer L4 with High-RAM |
| Drive storage full | Replaced mirror checkpoints can end up in the Drive trash, about 1 GB per run. Empty the trash |
| two notebooks at once | Never train the **same** run in two sessions on one Drive folder, because they would share `mirror/<run>/`. Split `MATRIX` into disjoint lists instead |
| `DEIM runtime modules missing` | The DEIM cell did not run or its install failed. Run `python train/deim_run.py --install-deps`, then `--check-deps` |
| `init checkpoint ... missing` | The COCO checkpoint is not in place. Run the DEIM cell again, or upload it to `MyDrive/ZanderML/pretrained/` (section 8) |
| `... sha256 ... != recorded` | A different init checkpoint than in earlier sessions. Delete the `.sha256` record only if that is intended |
| `LICENSE ... != pinned` | The pinned DEIM commit's licence no longer matches. Stop and re-check the licence; never just update the pin |
| `ONNX export does not match the torch model` | The parity gate caught an export problem. The training is safe in the mirror; report the numbers from the log |
| `DEIM training exited 0 but ...` | DEIM stopped early without an error code. The log above it says why; the run resumes from its last checkpoint |

Runtime knobs, set as environment variables before `train/matrix.py`:
- `ZANDER_MIRROR_INTERVAL_S` (default 300): the minimum number of seconds between mirror copies;
- `ZANDER_ABORT_AFTER_EPOCH=N`: a test hook that simulates a disconnect after epoch N.

## Plumbing check without a GPU (Mac)

```bash
cd ml
python train/dispatch.py --config configs/smoke_def3.yaml --device cpu        # one CPU epoch, then a bundle
ZANDER_ABORT_AFTER_EPOCH=1 python train/matrix.py --configs configs/smoke_resume_def3.yaml \
    --drive /tmp/zander-drive --device cpu --max-attempts 1                   # dies after epoch 1 (exit 1)
python train/matrix.py --configs configs/smoke_resume_def3.yaml --drive /tmp/zander-drive --device cpu
                                                                              # resumes -> bundle, 2 attempts
python train/matrix.py --configs configs/smoke_resume_def3.yaml --drive /tmp/zander-drive --device cpu
                                                                              # skipped
```

Use `--device cpu` on the Mac. Ultralytics training on MPS crashes with the pinned torch 2.5.1.

DEIM without a GPU:
```bash
cd ml
python train/deim_run.py --config configs/matrix/deim_s_def3_fold0.yaml --dry-run   # the full plan: clone, override, schedule, commands
python train/deim_run.py --check-deps                                                # DEIM's extra deps are NOT in ml/.venv (by design)
python train/dispatch.py --config configs/deim_smoke_def3.yaml --device cpu          # 2 CPU epochs, random init -> bundle
```
The CPU smoke needs DEIM's extra deps (faster-coco-eval, calflops, tensorboard) in the interpreter that runs it.
