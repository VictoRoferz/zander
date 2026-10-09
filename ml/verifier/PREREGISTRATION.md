# Verifier pre-registration (experiment E4, offline prototype)

Fixed on 2026-09-28, **before any real (non-smoke) verifier run**. The exit
report records this file's sha256; editing it after the first real run turns
every later run into an exploratory one. Every run is reported, including the
ones that fail.

## Question

Does a crop verifier behind the tiled detector find more *Nicht ausreichend
Lot* (NAL) defects at the operating range (≤ 0.5 false alarms per photo), or
remove NAL false alarms without letting a single extra defective photo
through — within the CPU budget?

## Scope (user decision 2026-09-28)

- Offline prototype. Not wired into the inspector; `detector.json.verifier`
  stays `null`.
- **NAL only is stacked.** Brücke and Fahne are passthrough classes (they keep
  the detector score): board B, which carries almost all of their boxes, is
  `train_always`, so they have no out-of-fold data. A class is stacked only
  with ≥ 30 positive out-of-fold candidates.
- Detector: the family chosen by the E1 selection. Its three fold models,
  each evaluated on its own fold at conf floor 0.001 (`<F>_fold0..2`), are the
  proposers. The OOF guard (`verifier/mine.py`) must pass without
  `--allow-non-oof`.
- Data: split_v2 `fold0..2` only. The `test` bucket is not touched;
  `probe_xprod`, `train_always` and `future` are not used.

## Frozen protocol

| item | value |
|---|---|
| candidates | top 50 per photo by detector score, conf floor 0.001 |
| labels | same-class IoU ≥ 0.3 → 1; best same-class IoU in [0.1, 0.3) → ignored; else 0 |
| configuration | `verifier/configs/mnv3s_2crop.yaml` as committed, no overrides, ImageNet-pretrained backbone |
| training | 15 fixed epochs, no early stopping, fixed seed; one run per family |
| cross-fit | nested (see `README.md`): stacker, tau and the detector threshold for fold k never see fold k |
| anchor | t* = the family's frozen operating point (`choose-op`, pooled out-of-fold photo sensitivity ≥ 0.95) |

## Endpoints

**Gate (c) — latency.** On the dev Mac (ONNX Runtime CPU FP32, default
threads, 20 photos after 2 warm-up photos, fold-0 detector model):
- verifier p50 ≤ **200 ms for 50 candidates** (crops + ONNX + stacker), and
- detector + verifier(50) p50 ≤ **1.0 s**.

**Primary (a).** Pooled over the three held-out folds: NAL recall at
0.5 FPPI (IoU 0.3, `eval/scorer.py`), cascade minus detector, on the same
photos. 95 % percentile interval from a paired cluster bootstrap over the
split_v2 board groups (1000 resamples, seed 0). **Success iff the lower bound
> 0.**

**Secondary (b).** Cross-fitted thresholds on every held-out fold k: the
detector at t(k) = the choose-op rule on the two other folds; the cascade at
tau(k), chosen on the same two folds as the highest tau that keeps every
detector-flagged positive photo flagged. Pooled over the three held-out folds:
**success iff the cascade has ≥ 30 % fewer NAL false-alarm boxes** (flagged,
unmatched at IoU 0.3) than the detector **and adds zero photo escapes**
(positive photos the detector flags and the cascade does not).

## Decision rule

**ship** iff (c) and ((a) or (b)); otherwise **no-ship**.

**invalid-smoke** (no decision at all) when any of: the candidates are not
out-of-fold (`--allow-non-oof`), fewer than all photos were mined
(`--fraction` < 1), the configuration differs from the committed one
(overrides such as `--epochs`, another config file), or the backbone fell
back to random initialisation.

"ship" only permits planning the inspector integration as a separate,
approved step (a `verifier` block in the detector card, inspector v2 with
conf floor 0.001 and top-50, latency re-measured on the Windows station).

## Descriptive only (no role in the decision)

(d) candidate ceiling (share of NAL GT boxes reachable in the top 50),
(e) inner vs outer out-of-fold AUROC, per-fold numbers, false-call rates on
seated and handheld iO photos, rescued photos.

## Known limitations, accepted up front

- Two endpoints joined by OR slightly inflate the false-positive rate of the
  decision (accepted by the user on 2026-09-28).
- Low power: about 81 positive photos across the dev folds. A null result
  ("no-ship") is the expected outcome unless the effect is large.
- The deployed detector would be the `dev_all` model, while the verifier and
  the stacker are fitted on fold-model candidates; the same approximation the
  detector's own t* already makes.
- Not allowed after seeing a report: choosing among several verifier runs by
  their exit numbers; re-tuning tau, hi-passthrough, C, epochs or the
  configuration on the folds; any test-bucket evaluation before a "ship" and
  the test ledger.

## Implementation corrections before the first real run, 2026-10-09

A code review found places where the code did not enforce what this file
already says. They were fixed **before any real (non-smoke) verifier run**. No
threshold, metric, endpoint or decision rule above changed; each item only
makes the code do what the text says.

- **Anchor = the family's frozen operating point** (Frozen protocol,
  "anchor"). `crossfit` matched the operating point by family only. It now
  also refuses one whose `target_sensitivity` differs from the pre-registered
  0.95 (or is missing) and one not frozen on exactly the three fold models
  that proposed the candidates (`sources[*].model_sha256`).
- **The test bucket is not touched** (Scope) and the cross-fit numbers are the
  out-of-fold ones. `cascade-eval` defaulted to `fold0`, where the final
  stacker, tau and verifier (fitted on fold0..2) are in-sample, recorded a
  results row there, and never ran the detector's eval guards. It now accepts
  only `test` / `probe_xprod` / `future`, runs `run_eval.check_eval_guards`
  (test: the dev_all detector only) and checks the detector's family.
- **Configuration "as committed, no overrides"** (Frozen protocol, Decision
  rule). Only the mine and crossfit configs were checked, and only by path.
  Now (1) the exit step's own config counts too: the gates are read from it,
  and `exit --set exit.latency_verifier_p50_ms=500 --set
  exit.latency_total_p50_s=2 --set exit.fa_reduction=0` had turned a no-ship
  into a ship; (2) "as committed" is the content: the sha256 of
  `configs/mnv3s_2crop.yaml` as committed with this file (1c28a8c, unchanged
  since) is pinned in `verifier/config.py`, so an in-place edit is no longer
  the default config; (3) uncommitted code under `ml/` (`git_dirty`, recorded
  at mine, crossfit and exit; the appended rows in `ml/reports/` excluded)
  makes the run exploratory. Any of these → invalid-smoke.
- **Gate (c) "default threads, 20 photos"**. `exit --threads N` and
  `--latency-photos N` were accepted silently. Now `--threads` other than 0
  (= ONNX Runtime's default intra-op threads) or a photo count other than the
  config's 20 → invalid-smoke. The bench model is picked as the fold-0 run by
  bucket, not by list position.
- **"Every run is reported"**. A second `exit` overwrote the first report.
  Now an existing report moves to `exit_archive/<UTC stamp>/`, and the new
  report and `stacker.json.exit.earlier_decisions` list every earlier one.
- **Secondary (b) "≥ 30 % fewer"**. `casc_fa <= (1 - 0.30) * det_fa` in
  floating point made exactly 30 % fail (0.7 × 90 = 62.999…); it is now
  compared exactly in rationals (10 × cascade ≤ 7 × detector).
- **"The backbone fell back to random initialisation"**. Only V_final's
  backbone was checked; a random-init fallback in any U_fold* / V_fold* model
  passed. Every model in `train_log.json` is checked now (a missing log or
  model is invalid too).
