"""
Score every fold bundle on ITS OWN validation fold — the Mac half of the
training matrix (Colab trains, the Mac evaluates).

    python cli.py eval-matrix                                          # ml_data/run_dirs/*_bundle.zip
    python cli.py eval-matrix --bundles "<Drive>/ZanderML/bundles/*_bundle.zip"

Per bundle zip, in sorted order:
  1. import it (ml_data/bundles/<run>/ + a run in MLflow experiment zander-v2);
  2. evaluate model.onnx on its own fold with the adapter and schema from
     meta.json, as eval run `<family>_fold<k>` — skipped when that run's
     metrics.json already exists for the same model sha (`--re-eval` forces);
  3. log the headline numbers into the bundle's MLflow run (`log_eval`).

dev_all bundles are skipped with a note: they have no out-of-fold bucket
(they get probe_xprod and the one-shot test after `choose-op`). Bundles the
eval guards refuse (e.g. a split_v1 bundle) are skipped with the reason; any
other failure is reported and the matrix goes on. Galleries are off by default
(12-17 MB each); smoke runs never enter results_v2.csv.
"""
from __future__ import annotations

import glob
import json
import logging
from pathlib import Path

import registry
from config.settings import settings
from eval import run_eval

logger = logging.getLogger("ml.eval_matrix")


def expand(patterns: list[str]) -> list[Path]:
    """Paths / glob patterns -> existing bundle zips (sorted, de-duplicated)."""
    out: dict[str, Path] = {}
    for pat in patterns:
        matches = glob.glob(str(pat)) if any(ch in str(pat) for ch in "*?[") else [str(pat)]
        for m in matches:
            p = Path(m)
            if p.exists():
                out[str(p.resolve())] = p
    return sorted(out.values(), key=lambda p: p.name)


def _cached(eval_run: str, model_sha: str) -> bool:
    mpath = settings.eval_dir / eval_run / "metrics.json"
    if not mpath.is_file():
        return False
    m = json.loads(mpath.read_text(encoding="utf-8"))
    cached_sha = (m.get("source") or {}).get("model_sha256")
    if cached_sha != model_sha:
        logger.warning(f"{eval_run}: existing metrics are for another/unknown model — re-evaluating")
        return False
    return True


def run(
    bundles: list[str] | None = None,
    re_eval: bool = False,
    gallery: bool = False,
    record: bool = True,
    force_import: bool = False,
    mlflow: bool = True,
) -> list[dict]:
    paths = expand(bundles or [str(settings.run_dirs_dir / "*_bundle.zip")])
    if not paths:
        raise FileNotFoundError(f"no bundle zips match {bundles}")
    rows = []
    for zp in paths:
        row = {"bundle": zp.name, "run_name": None, "eval_run": None, "status": None, "note": ""}
        try:
            imp = registry.import_bundle(zp, force=force_import, register=mlflow)
            info = imp["info"]
            row["run_name"] = info["run_name"]
            fold = info["fold"]
            if fold == "dev_all":
                row.update(status="skipped", note="dev_all bundle: no out-of-fold bucket — "
                           "evaluate probe_xprod + the one-shot test with the frozen op")
            elif not (fold or "").startswith("fold"):
                row.update(status="skipped", note=f"unknown fold {fold!r} in meta.json")
            else:
                eval_run = f"{info['family']}_{fold}"
                row["eval_run"] = eval_run
                model = imp["bundle_dir"] / "model.onnx"
                if not re_eval and _cached(eval_run, registry.file_sha256(model)):
                    row["status"] = "cached"
                else:
                    m = run_eval.run(
                        model_path=model, adapter_name=info["adapter"] or "ultralytics",
                        schema=info["schema"], bucket=fold, run_name=eval_run,
                        record=record, gallery=gallery,
                    )
                    row["status"] = "evaluated"
                    row["note"] = f"{m['n_photos']} photos, {m['mean_ms_per_photo']} ms/photo"
                if mlflow:
                    registry.log_eval(eval_run)
            if info["bundle_version"] < 2:
                row["note"] = (row["note"] + "; " if row["note"] else "") + "bundle v1"
        except (run_eval.EvalGuardError, registry.BundleError) as e:
            row.update(status="skipped", note=str(e))
        except Exception as e:  # keep the matrix going; report at the end
            logger.exception(f"{zp.name}: failed")
            row.update(status="failed", note=f"{type(e).__name__}: {e}")
        logger.info(f"[{row['status']}] {zp.name} {row['eval_run'] or ''} {row['note']}")
        rows.append(row)

    width = max(len(r["bundle"]) for r in rows)
    lines = ["", "eval-matrix summary:"] + [
        f"  {r['status']:<9} {r['bundle']:<{width}}  {r['eval_run'] or '-':<28} {r['note'][:160]}"
        for r in rows
    ]
    logger.info("\n".join(lines))
    return rows
