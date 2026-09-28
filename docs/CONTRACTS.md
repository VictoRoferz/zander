# Zander interface contracts (v1, 2026-09-28)

The shared interfaces between camerapi, receiver, inspector, dashboard and the ML toolchain. Every work package codes against this file. Changing a contract means changing this file first, in its own commit.

Paths are relative to the repo root (`zander/`). `DATA_ROOT` is the hub data folder (`/data` in Docker, `deploy\data` on the Windows host, `~/zander-data` natively).

---

## 1. `detector.json` v1: the exported detector's card

Written by `cli.py export`, read by the inspector. Both go through **`ml/infer/detector_card.py`** (`write_detector_card`, `load_detector_card`, `validate_card`), so both sides run the same validator. The card sits next to the model and shares its stem: `detector.onnx` + `detector.json`.

```json
{ "contract_version": 1, "kind": "zander-detector",
  "model_file": "detector.onnx", "model_sha256": "<64 hex>",
  "adapter": "ultralytics", "schema": "def3",
  "class_names": ["Nicht ausreichend Lot", "Brücke", "Fahne"],
  "threshold": 0.3412, "conf_floor": 0.3412,
  "decision_rule": "NiO iff any detection score >= threshold (inclusive); classes = those with >= 1 such detection",
  "tiler_version": "v2", "tile_size": 640, "stride": 512,
  "operating_point": {"file": "...", "sha256": "...", "family": "yolo26n_def3", "rule": "...", "target_sensitivity": 0.95, "oof_level1": {}},
  "source": {"bundle_run": "yolo26n_def3_devall", "fold": "dev_all", "weights": "last", "bundle_sha256": "...",
             "dataset_id": "ds_v2_def3", "dataset_fingerprint": "...", "split_id": "split_v2", "snapshot_id": "...",
             "git_sha": "...", "framework": "ultralytics", "versions": {}},
  "validation": {"test_eval_run": "yolo26n_def3_devall__test", "test_level1": {}, "parity": {"photos": 5, "verdict_mismatches": 0}},
  "verifier": null,
  "exported_at": "...Z", "exported_by": "zander-ml 0.1.0 cli export" }
```

**Validation (enforced by `validate_card`):**
- `contract_version == 1` and `kind == "zander-detector"`.
- `model_file` is a bare file name, and the card must name the model it is loaded with.
- `model_sha256` has 64 hex characters and matches the file (the reader can skip this only explicitly, with `verify_sha=False`).
- `adapter` is one of `ultralytics` or `deim`.
- `class_names` is a non-empty list of unique strings.
- `0 < threshold < 1` and `0 < conf_floor < 1`.
- If `verifier` is null, `conf_floor <= threshold` (export writes them equal). Otherwise `conf_floor == verifier.candidate_floor`.
- `tiler_version`, `tile_size` and `stride` equal `infer.tiling`'s constants.

**Why `conf_floor == threshold` is safe without a verifier.** Merging is greedy by descending score, and a kept box keeps its maximum score. So the verdict, the class set and `photo_score` are identical at floor t and at 0.001. Only box outlines are tighter.

**Inspector behaviour:**
- **The card is authoritative.** When a card exists, `ADAPTER`, `CONFIDENCE_THRESHOLD` and `CLASS_NAMES` from the environment are ignored, with a WARNING.
- **No card:** the legacy env settings are used, with a WARNING; `threshold_source = "env"` or `"default"`. If `REQUIRE_CARD=true`, the inspector refuses to load instead.
- **Refuses to load** (the service stays up, `/inspect` returns 503, and `/status.load_error` explains why) when:
  - the card is invalid;
  - the sha doesn't match;
  - the class count the adapter reports differs from `len(class_names)`;
  - `verifier` is non-null. That reserved shape is `{onnx, onnx_sha256, stacker, stacker_sha256, top_k, candidate_floor}`, and inspector v1 has no verifier support.
- Builds `TiledDetector(conf_floor=card.conf_floor)`; today `detector_service.load` passes the threshold as the floor.

## 2. `pred.json` v1: one prediction per capture

**Location:** `DATA_ROOT/predictions/<capture_id>.pred.json`. It has its own folder so that:
- no existing glob matches it (`unlabeled/*.jpg`, `<id>.json`), and Label Studio's local storage ignores it;
- handover folders stay free of model output;
- it can be wiped when a new model arrives.

**Writing rules (receiver only):**
- Atomic write through `_atomic_write_bytes`: temp file → fsync → `0644` → `os.replace`.
- UTF-8 with `ensure_ascii=False`.
- `pending` is never written; the dashboard derives it.
- A file with `status:"ok"` is only replaced by another `ok` from a different `model_version`. It is never downgraded to `no_model` or `error`.

| field | type | notes |
|---|---|---|
| `schema_version` | int | always `1` |
| `capture_id` | str | |
| `status` | `"ok"` \| `"no_model"` \| `"error"` | |
| `created_at` | str | ISO-8601 UTC with `Z`; the time the receiver wrote the file |
| `error` | str \| null | max 500 chars; null when `ok` |
| `model_version` | str \| null | `"<file>@<sha256[:8]>"`, the idempotency key |
| `model_sha256`, `schema`, `tiler_version` | str \| null | |
| `class_names` | list[str] \| null | in the model's output order |
| `threshold` | float \| null | the t* used for the verdict and the boxes |
| `threshold_source` | `"detector.json"` \| `"env"` \| `"default"` \| null | |
| `image_width`, `image_height` | int \| null | the pixel frame of all boxes |
| `verdict` | `"iO"` \| `"NiO"` \| null | null unless `ok` |
| `classes` | list[str] | `[]` unless `ok`; in schema order |
| `photo_score` | float \| null | the highest detection score |
| `detections` | list of `{x1,y1,x2,y2: float, score: float, label: str}` | photo-pixel xyxy, highest score first, all ≥ `threshold`; `[]` unless `ok` |
| `latency_ms` | float \| null | inference time inside the inspector |

## 3. Inspector API (`:8004`)

`POST /api/v1/inspect` takes a multipart `file`, or a form field `capture_id` (resolved to `DATA_ROOT/unlabeled/<id>.jpg`).

- **Response 200** keeps today's fields: `status, source, verdict, classes, detections, count, latency_ms, model_version, tiler_version, confidence_threshold`.
- It adds: `photo_score, image_width, image_height, model_sha256, schema, adapter, class_names, threshold_source`.
- **Errors:**
  - 503 `{"detail": "no model loaded: <load_error>"}`
  - 422 for a missing, empty or undecodable image, or an invalid capture_id
  - 404 for an unknown capture_id
  - 500 `"inference failed: …"`
- **Execution:** inference runs in the threadpool (`run_in_threadpool`), serialized by a lock. The image is decoded outside the lock.

`GET /api/v1/status` returns: `service, version, model_loaded, model_path, meta_path, model_version, model_sha256, adapter, schema, class_names, confidence_threshold, threshold_source, tiler_version, load_error`, plus `meta` (`split_id, dataset_fingerprint, run_name, created`, taken from the card's `source` and `exported_at`).

`GET /api/v1/health` returns `{"status":"ok"}` even when no model is loaded, so the container stays healthy.

## 4. camerapi live preview (`:8001`)

The preview is switched on with `LIVE_PREVIEW_ENABLED`, which is **false by default**. When it is false, everything behaves exactly as today (one-shot `GrabOne` capture).

**Endpoints (all behind the access rule below):**
- `GET /api/v1/preview/stream.mjpg`
  - `Content-Type: multipart/x-mixed-replace; boundary=frame`.
  - Each part: `--frame\r\nContent-Type: image/jpeg\r\nContent-Length: N\r\n\r\n<jpeg>\r\n`.
  - The last frame, or a placeholder, is re-sent at least every 2 s.
- `GET /api/v1/preview/frame.jpg`: one fresh preview JPEG. Waits up to 3 s and starts the stream if it is idle. `Cache-Control: no-store`.
- `GET /api/v1/preview/status`: returns `enabled, state, source, camera_model, camera_serial, profile, fps_target, fps_measured, clients, frame_seq, last_frame_age_s, frames_incomplete, reconnects, last_error, preview_width, preview_height, boot_id`.
  - `state` is one of `disabled`, `starting`, `idle`, `streaming`, `reconnecting`, `unavailable` (Wi-Fi profile), `stopped`.
  - When disabled it returns 200 with `enabled:false`.
- Stream and frame return 404 when the preview is disabled, and 503 when `PREVIEW_MAX_CLIENTS` is reached.
- `GET /api/v1/status` gains a `live` block with the same fields as preview status.

**Access rule.** A request is allowed if:
- `PREVIEW_TOKEN` is set and the `X-Preview-Token` header equals it (`hmac.compare_digest`); or
- `PREVIEW_TOKEN` is empty, the peer address is loopback, and the `Host` header is `localhost`, `127.0.0.1` or `[::1]`.

Otherwise the answer is 401 (token missing or wrong) or 403 (not local and no token configured). Production uses the token, because Docker→host traffic is not guaranteed to arrive as loopback. The browser never sees the token: the dashboard proxies the stream.

**Capture semantics.** `POST /api/v1/capture` keeps its contract (202, the same `CaptureResult`). With the preview on, the camera-owner thread stops the stream and does a `GrabOne`, so the capture is the **first frame after the trigger**. It then resumes streaming. During an outage the capture falls back to `sample.jpg`, as today, and the response reports `source:"fallback"`.

**Env vars:**

| var | default |
|---|---|
| `LIVE_PREVIEW_ENABLED` | `false` |
| `PREVIEW_TOKEN` | `""` |
| `PREVIEW_FPS` | `5` |
| `PREVIEW_MAX_WIDTH` | `1280` |
| `PREVIEW_JPEG_QUALITY` | `70` |
| `PREVIEW_IDLE_STOP_S` | `10` (stream stops this long after the last viewer leaves) |
| `PREVIEW_MAX_CLIENTS` | `8` |
| `CAMERA_RECONNECT_MAX_BACKOFF_S` | `30` |
| `LIVE_GRAB_TIMEOUT_MS` | `5000` |
| `LIVE_CAPTURE_TIMEOUT_S` | `20` |

Log strings stay ASCII (Windows cp1252 consoles).

## 5. Dashboard API (`:8003`, all endpoints need a logged-in session)

- `GET /api/latest` and `GET /api/captures/{capture_id}` return the same shape. The id must match `^[A-Za-z0-9_-]{1,64}$` (400 otherwise; 404 if unknown).
```json
{"capture": null | {"capture_id","captured_at","triggered_by","source",
                    "image_url": "/thumb/unlabeled/<id>.jpg","width","height"},
 "state": "pending"|"ok"|"no_model"|"error"|"none",
 "prediction": null | <pred.json as stored>}
```
  - `width`/`height` come from `prediction.image_*`, falling back to the capture sidecar.
  - `state` is the file's `status` when a prediction file exists. Without one it is `pending` while the photo is younger than `PREDICTION_PENDING_TIMEOUT_S`, and `none` after that.
  - A `schema_version > 1` gives `state:"error"` ("unsupported").
- `GET /api/live.mjpg` relays camerapi's stream and adds `X-Preview-Token`. Errors: 401, 404 (preview disabled), 502 (camerapi unreachable or token rejected; logged), 503 (too many viewers).
- `GET /api/live/status` returns `{"reachable": bool, "error": str|null, ...camerapi preview status}`. camerapi 401/403 maps to `state:"auth_error"`; a 404 from an old camerapi maps to `state:"unsupported"`.
- **Env:** `PREVIEW_TOKEN` (`""`), `PREDICTION_PENDING_TIMEOUT_S` (`60`), `PREDICTIONS_DIR` (default `DATA_ROOT/predictions`, read-only).

## 6. Receiver prediction hook (`:8002`)

- **Env:**
  - `INSPECTOR_URL` (`""` turns predictions off; compose sets `http://inspector:8004`; `launch.py` injects `http://localhost:8004` only when it started the inspector itself).
  - `INSPECTOR_TIMEOUT` (`60`), `INSPECTOR_STATUS_TIMEOUT` (`5`), `PREDICTION_QUEUE_MAX` (`1000`).
- **Queue:**
  - `prediction_service.enqueue(capture_id, force=False) -> bool` never blocks.
  - Ingest calls it after `write_task_marker`, and in the dedupe branch only when no `pred.json` exists. Both calls are wrapped so the ingest ACK is never affected.
  - One worker thread (the `spool_uploader` pattern) handles one inspector request at a time. The queue is bounded and deduped by id.
- **`process_one(capture_id)`:**
  1. `GET /status`. If unreachable, write `error` ("inspector unreachable"). If `model_loaded:false`, write `no_model` with the `load_error`.
  2. Skip if an existing `ok` has the same `model_version` (unless `force`).
  3. `POST /inspect` with the file and the `capture_id`:
     - 200 → validated `ok` document (`confidence_threshold` maps to `threshold`);
     - 503 → `no_model`;
     - anything else → `error` "HTTP <code>: <detail>".
  4. Never overwrite an existing `ok` with a non-`ok` result.
- **Optional:** `POST /api/v1/predict/{capture_id}?force=false` returns 202 `{"queued": bool}`, or 404 if there is no image.
- `/api/v1/status` gains `predictions: {enabled, inspector_url, queue_depth, count}`.

## 7. Deploy

**`deploy/docker-compose.yml`:**
- `receiver` gets `INSPECTOR_URL: ${INSPECTOR_URL-http://inspector:8004}`. Use `-`, not `:-`, so that an empty value disables it. Also `INSPECTOR_TIMEOUT`. There is **no** `depends_on` for the inspector.
- `dashboard` gets `PREVIEW_TOKEN: ${PREVIEW_TOKEN:-}` and `PREDICTION_PENDING_TIMEOUT_S`.
- New `inspector` service:
  - `build: {context: .., dockerfile: inspector/Dockerfile}`, `container_name: zander-inspector`
  - ports `127.0.0.1:8004:8004`
  - env `DATA_ROOT=/data`, `MODEL_PATH=/data/models/detector.onnx`, `ORT_THREADS=${INSPECTOR_ORT_THREADS:-0}`
  - volume `${DATA_DIR:-./data}:/data:ro`
  - `restart: unless-stopped`
  - healthcheck: curl `/api/v1/health`

**Root `.dockerignore`** (an allowlist; only the inspector builds from the repo root):
```
*
!inspector/
!ml/infer/
**/__pycache__
**/*.pyc
inspector/.env
```

**`scripts/launch.py`:**
- In hub modes it starts the inspector as a **non-critical** child. The interpreter is `INSPECTOR_PYTHON`, else `ml/.venv` (checked with `-c "import onnxruntime, PIL, fastapi"`). If none works it logs "inspector skipped". `--no-inspector` turns it off.
- A non-critical child that exits is logged once and never stops the hub.
- camerapi still gets only `base_env`.

## 8. ML toolchain

### 8.1 Bundle v2 (written by `train/*`)

- `<run>_bundle.zip` is flat:
  - `config.yaml` (the **effective** merged config)
  - `model.onnx`, plus `model_raw.onnx` for YOLO26 (the raw head)
  - `last.pt` (or `best.pt`)
  - `metrics.json`, `pip-freeze.txt`, `meta.json`
  - `LICENSE_DEIM` for DEIM bundles
- `meta.json` = all v1 keys plus:
  - `bundle_version: 2`, `family`, `weights: "last"|"best"`, `epochs`, `epochs_completed`, `train_seconds`
  - `attempts: [{gpu, started, seconds, resumed}]`, `config_sha256`, `overrides`
  - `onnx: {file, sha256, opset, layout: raw|e2e|deim, output_shapes}`, `onnx_raw: {...}|null`
  - `versions: {python, torch, cuda, ultralytics | deim_commit}`, `created`
- **Naming:**
  - Config `name`: `<arch>_<schema>_fold<k>` or `<arch>_<schema>_devall`.
  - Family: `<arch>_<schema>`, e.g. `yolo26n_def3`.
  - Readers treat a missing `bundle_version` as 1 and derive `family` from `run_name`.
- **Entry point:** `train.dispatch.run(config_path: Path, device: str|None = None, overrides: dict|None = None, mirror_dir: Path|None = None) -> Path` returns the bundle zip. Module CLIs: `python train/dispatch.py`, `python train/matrix.py`.
- **DEIM:** `train.deim_run.run_from_config(config_path, device=None, overrides=None, mirror_dir=None) -> Path` returns the same bundle format.

### 8.2 Eval runs, `predictions.json` v2, frozen eval API

- **Run names:**
  - own fold: `<family>_fold<k>` (= the bundle run name)
  - other buckets: `<bundle_run>__<bucket>`
  - cascade: `<family>+<vrun>_fold<k>`
  - A family name resolves to `<F>_fold0..2`.
- **`predictions.json` v2** = v1 plus:
  - `format_version: 2`
  - `source: {kind: detector|cascade, bundle_run, family, fold, model_sha256, model_resolved, adapter, conf_floor, verifier_run, stacker_sha256}`
  - per-photo `width`, `height`
  - `det_scores` (cascade runs only)
  - `scores` is always what the decision uses. Loaders accept v1.
- **Frozen API.** Keyword-only additions are allowed; renames, removals and reorders are not:
  - `run_eval.load_predictions`, `bucket_photos`, `_gt_record`, `_write_outputs(out_dir, metrics, photos)`, `format_report`
  - `photo_scorer.score_all`, `threshold_for_sensitivity`, `class_records`, `match_indices`
  - all of `scorer.*`
  - `compare.run(run_a, run_b, class_index=0, seed=0)`
- `_write_outputs` needs these `metrics` keys: run_name, model, adapter, schema, bucket, threshold, threshold_source, n_photos, mean_ms_per_photo, tiler_version, split_id, snapshot_id, dataset_fingerprint, host.
- The `ml/reports/results_v2.csv` columns are unchanged. Smoke rows are never committed.

### 8.3 `operating_point.json` v2 (`choose-op`)

- All v1 keys (`schema, class_names, threshold, rule, runs, models, split_id, created, oof_level1`), plus `family`, `target_sensitivity` and `sources: [{eval_run, bundle_run, fold, model_sha256, config_sha256, dataset_fingerprint}]`.
- File: `ml_data/eval/operating_point_<schema>_<family>.json`.
- It is frozen: overwriting needs `--force`, and the old file moves to `ml_data/eval/archive/`.

### 8.4 Verifier artifacts and `ml/infer/verifier.py`

- **Crop geometry**, one function for training and inference:
  - Constants: `WINDOW=160, SMALL=64, LARGE=128, SCALE_REF=48, JITTER=8`.
  - `s = max(1, max(w,h)/48)`. Cut a square of side `round(160·s)` centred on the box, from the full-resolution RGB photo. Reflect-pad only the overhang. Resize to 160 (PIL bilinear) only if `s > 1`.
  - `large = win[16:144,16:144]`, `small = win[48:112,48:112]`. Training shifts both crops by the same `(dx,dy)` in [−8,8]².
- **`verifier.onnx`:**
  - Inputs `crop_small` float32 [N,3,64,64] and `crop_large` float32 [N,3,128,128]. RGB, 0..1; ImageNet normalisation is inside the graph.
  - Outputs `logits` [N,C] and `embedding` [N,128].
  - Opset 17, dynamic N.
  - Network: MobileNetV3-Small features (shared) → global pool, 576 per crop → concat 1152 → Linear 1152→128 → Hardswish → Dropout 0.2 → Linear 128→C.
- **`stacker.json`:**
  - `contract_version: 1, kind: "zander-stacker"`, `schema`, `class_names`, `verifier_file`, `verifier_sha256`
  - `top_k: 50, candidate_floor: 0.001`, `detector_threshold`, `crop{...}`
  - `features: ["det_logit","ver_logit","log_area","seam_margin"]`
  - `classes: {<name>: {"mode":"stack", mean, std, coef, intercept, tau, hi_passthrough} | {"mode":"passthrough"}}`
  - `anchor: "piecewise_linear"`, `knn: null`, `provenance{...}`, `exit{decision: ship|no-ship|invalid-smoke, report}`
- **Features:**
  - `det_logit = logit(clip(s,1e-6,1-1e-6))`
  - `ver_logit` = the logit of the candidate's class head
  - `log_area = log(max(w·h,1))`
  - `seam_margin = min(128, max over tiles fully containing the box of the distance to that tile's interior edges)/128`; 0 if no tile contains the box
  - `p = σ(intercept + coef·(x−mean)/std)`
- **Anchored final score** (keeps `verdict.py`'s single threshold):
  - `A(p) = t*·p/τ` for `p < τ`, and `t* + (1−t*)(p−τ)/(1−τ)` for `p ≥ τ`.
  - Passthrough classes keep the detector score.
  - `hi_passthrough h`: final = `max(A(p), s_det)` when `s_det ≥ h`.
- **`ml/infer/verifier.py`** (numpy, onnxruntime and Pillow only; importing it must not load torch):
  `extract_window(photo_rgb, box)`, `crops_from_window(win, dx=0, dy=0)`, `seam_margin(box, width, height)`, `anchor_map(p, tau, t_star)`, `class Stacker` (`from_dict`, `features`, `proba`, `final`), `class Verifier(stacker_path, threads=0)` with `rescore(photo_rgb, det: Detections) -> Detections`.
- **Plug-in point:** `TiledDetector(conf_floor=0.001).detect(rgb)` → `verifier.rescore(rgb, det)` → `photo_verdict(threshold=t*)`.
- **Working files** (internal to WP-verifier), in `ml_data/verifier/<vrun>/`: `candidates.json`, `windows.npy` (+ index), `oof_scores.json`, `verifier.onnx`, `stacker.json`, `exit_report.{md,json}`, `train_log.json`.
- **Entry point:** `verifier.cli.main(argv: list[str]) -> int`; module CLI `python verifier/cli.py`.

### 8.5 Colab and Drive layout

`MyDrive/ZanderML/`:
- `ds_v2_{def3,nal1}_tiles.zip`: input.
- `bundles/<run>_bundle.zip` plus `.sha256`: the "finished" marker.
- `mirror/<run>/{last.pt, results.csv, args.yaml}`: for resume; deleted once the bundle is safe.
- `logs/<run>.log`.
- `matrix_log.jsonl`.

## 9. Tests

All tests are plain assert scripts run from the repo root with `-B`:
- ML and inspector: `ml/.venv/bin/python -B tests/test_<x>.py`
- camerapi, receiver, dashboard: `../venv/bin/python -B tests/test_<x>.py`

Each test file loads **one** service. The services share top-level module names (`config`, `services`, `api`, `models`, `main`). Set `DATA_ROOT` to a temp folder and set every relevant env var before importing, because `.env` is read from the current directory and settings create folders at import.
