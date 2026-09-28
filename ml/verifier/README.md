# verifier — the second stage of the cascade (offline prototype, NAL only)

The tiled detector proposes; a small network re-reads each of the top 50
candidates on crops cut from the full-resolution photo; a logistic stacker
fuses both. Contracts: `docs/CONTRACTS.md` §8.2 and §8.4. Exit rule:
`PREREGISTRATION.md`. Design background: the architecture page (sections
"Recommended architecture" and "Inside the models").

```
photo -> TiledDetector(conf_floor 0.001) -> top 50 by detector score
      -> NAL candidates: 160-px window -> crops 64 + 128 -> verifier.onnx -> ver_logit
         stacker p = sigmoid(b + w . std([det_logit, ver_logit, log_area, seam_margin]))
         final = A(p): tau -> t* (anchored)   [hi-passthrough keeps strong detections]
      -> Brücke / Fahne: passthrough (detector score)
      -> infer/verdict.py with the ONE threshold t*
```

## Commands (from `ml/`)

```bash
python verifier/cli.py all --family yolo26n_def3 --run-name v1 --device mps
# or step by step:
python verifier/cli.py mine --family yolo26n_def3 --run-name v1
python verifier/cli.py crops --run-name v1
python verifier/cli.py crossfit --run-name v1 --device mps   # op: ml_data/eval/operating_point_<schema>_<F>.json
python verifier/cli.py export --run-name v1
python verifier/cli.py exit --run-name v1
# detector + verifier on any bucket (test refused until run_eval.check_test_ledger exists):
python verifier/cli.py cascade-eval --run-name v1 --model <detector.onnx> --bucket fold0
```

`python cli.py verifier ...` (ml/cli.py) delegates to `verifier.cli.main(argv)`.
Set `PYTORCH_ENABLE_MPS_FALLBACK=1` on a Mac. The ImageNet weights come from
torchvision's cache (`TORCH_HOME`); on a python.org build without a CA bundle
the download is retried with `certifi`, and a failed download falls back to
random init (reported; the run can then never ship).

Smoke run (no GPU; the three smoke fold runs share one model, so they are not
out-of-fold and the report says `invalid-smoke`):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python verifier/cli.py all \
  --runs smoke_def3_fold0 smoke_def3_fold1 smoke_def3_fold2 --allow-non-oof \
  --run-name smoke_v --device mps --fraction 0.2 --epochs 1 \
  --operating-point ../ml_data/eval/operating_point_def3_smoke_def3_fold0.json
```

## Method

1. **Mining** (`mine.py`). Out-of-fold candidates are the detections each
   fold model made on its own held-out fold (standard stacking
   cross-validation). The OOF guard refuses anything else. Top 50 per photo;
   labels by same-class IoU (≥ 0.3 positive, [0.1, 0.3) ignored), negatives
   typed background / other_defect / wrong_class. Every NAL GT box is added
   once as an extra positive for verifier training only.
2. **Windows** (`crops.py`). One 160-px window per candidate, cut once with
   `infer.verifier.extract_window`, the function the runtime uses:
   s = max(1, max(w, h)/48), a round(160 s) square, reflect padding on the
   overhang only, PIL bilinear to 160 only if s > 1. Crops: centre 128 and 64.
3. **Network** (`model.py`). MobileNetV3-Small (ImageNet) shared by both
   crops, 576 + 576 pooled features, Linear 1152→128, Hardswish, Dropout 0.2,
   one logit per class. BatchNorm statistics stay frozen (the two crop sizes
   would otherwise mix their batch statistics).
4. **Training** (`train.py`). Masked BCE on the candidate's class head;
   25 % positives per epoch; per sample the same random shift (±8 px) and D4
   element for both crops, cut by `crops_from_window`; brightness / contrast
   / saturation / hue jitter shared by both crops; AdamW, cosine schedule,
   15 fixed epochs, no early stopping.
5. **Nested cross-fit** (`crossfit.py`), below.
6. **Stacker** (`stacker.py`). Numpy Newton/IRLS logistic regression on
   standardised [det_logit, ver_logit, log_area, seam_margin], L2 with C = 1,
   intercept unpenalised. tau = the highest value that keeps every positive
   photo the detector flags at t flagged. The stacker probability is mapped
   onto the detector scale with A(tau) = t*, so `verdict.py` keeps one
   threshold and A(p) ≥ t* exactly when p ≥ tau. Hi-passthrough (auto): above
   the highest-scoring negative, if at least 5 positives lie there, detections
   keep at least their detector score.
7. **Export** (`export_onnx.py`). V_final to `verifier.onnx` (opset 17,
   dynamic N, parity ≤ 1e-4 on 32 cached windows) plus `stacker.json`.
8. **Exit** (`exit_rule.py`), as pre-registered, into `exit_report.{md,json}`
   and `stacker.json.exit`.

## Why nested

The stacker learns how far to trust the verifier. Fitted on scores the
verifier produced for its own training data, it would learn to trust an
overconfident, in-sample verifier, and the cascade's cross-validated number
would be optimistic. So for every outer fold k with inner folds {i, j}:

| step | trained / chosen on | applied to |
|---|---|---|
| verifier U_j, U_i | one inner fold each | the other inner fold (inner OOF scores) |
| stacker S_k, detector threshold t(k), tau(k) | inner OOF scores of i and j | fold k |
| verifier V_k | i ∪ j | fold k |
| cascade score on k | S_k(V_k) | the written eval run `<F>+<vrun>_fold<k>` |

Nothing used on fold k has seen fold k (`crossfit.check_plan`,
`tests/test_verifier_mining.py`). A single-fold verifier is the same model
whichever outer fold uses it, so the six inner models are three trainings.
Total: 3 single-fold + 3 two-fold + 1 final = 7 trainings. The final
artifacts are V_final (all folds), S_final (fitted on the pooled outer OOF
scores) and tau_final (pooled, at t*). The cost of nesting is a mismatch: S_k
is fitted on scores of verifiers trained on a third of the data and applied
to one trained on two thirds. Exit item (e) reports it as inner versus outer
AUROC.

## Differences from the report's design

- **Proposers.** The report's plan mines candidates with *auxiliary* stage-1
  models ("train stage 1 on two folds, propose on the unseen third"). Here the
  selection matrix's own fold models are the proposers: no extra detector
  training, and the candidates come from exactly the detector that was
  evaluated.
- **Stacker.** The report does not fix how the stacker is fitted. Here it is
  nested (above). Only fusion is implemented: no hard gating, which cut AP in
  one MIDOG entry.
- **Not implemented.** The normal-joint kNN feature (2b, `knn: null`),
  ResNet-18 and 64-only ablations, and per-class verification for
  Brücke/Fahne (no OOF data).
- **Remaining mismatch.** The deployed detector is the `dev_all` model. The
  verifier and stacker were fitted on fold-model candidates, so the score
  distributions differ somewhat. The detector's own t* makes the same
  approximation.

## Artifacts (`ml_data/verifier/<vrun>/`)

`candidates.json` · `windows.npy` + `windows_index.json` (pins the candidates
sha) · `oof_scores.json` (per-candidate inner/outer scores, per-fold t(k),
tau(k), stackers) · `stacker_fold<k>.json` (S_k, referenced by the cascade
runs' `source.stacker_sha256`) · `verifier_final.pt` · `verifier.onnx` ·
`stacker.json` · `exit_report.{md,json}` · `train_log.json`.
Cascade eval runs go to `ml_data/eval/<F>+<vrun>_fold<k>/` (predictions.json
v2, `source.kind = cascade`). Smoke and fractional runs never append to
`ml/reports/results_v2.csv`.
