"""
Class-aware photo-level evaluation of one ONNX checkpoint on one bucket.

    python cli.py eval --model m.onnx --schema def3 --bucket fold0
    python cli.py choose-op <run_fold0> <run_fold1> <run_fold2>
    python cli.py eval --model final.onnx --bucket test --operating-point op.json

Buckets (split_v2, dataprep/split.py):
  fold0|fold1|fold2  validation fold of the model trained without it
  probe_xprod        photos holding the board-A Brücke/Fahne probe boxes
                     (per-box hits only — their other defects may be trained)
  train_always       board B + A2 (always trained): memorisation check only
  test               ONE-SHOT, final model only
  future             captures after the split freeze (chronological test)

Honesty rules enforced here:
  - test / probe_xprod / future REQUIRE --operating-point (a threshold frozen
    on out-of-fold dev predictions); the threshold is never re-derived there.
  - fold runs without an operating point report a self-tuned threshold,
    flagged as diagnostic; `choose-op` pools the three fold runs' predictions
    into the real dev operating point.

Writes ml_data/eval/<run_name>/{metrics.json, predictions.json, report.md,
gallery.html} and appends one row to the committed ml/reports/results_v2.csv.
Model-vs-model decisions use ONLY these photo-level numbers — never per-tile
or in-training validation metrics.
"""
from __future__ import annotations

import csv
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from dataprep.ingest import load_manifest, photo_path, usable
from dataprep.remap import SHORT, project, schema_classes
from dataprep.split import bucket_of, load_split
from eval.gallery import write_gallery
from eval.photo_scorer import match_indices, score_all, threshold_for_sensitivity
from eval.scorer import FPPI_OPERATING, IOU_OPERATING, iou
from infer.pipeline import TiledDetector

logger = logging.getLogger("ml.eval")

FROZEN_OP_BUCKETS = {"test", "probe_xprod", "future"}
NOT_HONEST_BUCKETS = {
    "probe_xprod": "only the per-box probe section is honest here — these photos' OTHER "
                   "defects may be in the model's training data; levels 1-3 are context only",
    "train_always": "board B is always trained: every number below is memorisation, not validation",
}
RESULTS_FILE = "results_v2.csv"
RESULTS_FIELDS = [
    "date", "run_name", "model", "adapter", "schema", "bucket", "threshold",
    "threshold_source", "n_photos", "n_positive", "n_negative_seated",
    "photo_auroc", "escape", "escape_ci_lo", "escape_ci_hi",
    "fcr_seated", "fcr_seated_ci_lo", "fcr_seated_ci_hi", "fcr_handheld",
    "recall_nal_0.5fppi", "recall_nal_ci_lo", "recall_nal_ci_hi",
    "recall_bruecke_0.5fppi", "recall_fahne_0.5fppi", "recall_any_0.5fppi",
    "exact_set_acc", "ap50_nal", "probe_found", "mean_ms_per_photo",
    "tiler_version", "split_id", "snapshot_id", "dataset_fingerprint", "host",
]


def bucket_photos(bucket: str, schema: str) -> tuple[list[dict], list[dict]]:
    """(manifest entries of the bucket, probe boxes relevant to it)."""
    manifest, split = load_manifest(), load_split()
    classes = schema_classes(schema)
    probe = [p for p in split["probe_boxes"] if p["class"] in classes]
    probe_ids = {p["capture_id"] for p in probe}
    entries = []
    for e in manifest["entries"]:
        if not usable(e):
            continue
        b = bucket_of(e, split)
        if bucket == "probe_xprod":
            if e["capture_id"] in probe_ids:
                entries.append(e)
        elif b == bucket:
            entries.append(e)
    if not entries:
        raise ValueError(f"no usable photos in bucket {bucket!r} (schema {schema})")
    ids = {e["capture_id"] for e in entries}
    return entries, [p for p in probe if p["capture_id"] in ids]


def _gt_record(e: dict, schema: str, split: dict) -> dict:
    train, other = project(e["boxes"], schema)
    return {
        "capture_id": e["capture_id"],
        "group": split["assignment"].get(e["capture_id"], {}).get("group", e["capture_id"]),
        "board": e["board"],
        "pose": e["pose"],
        "gt_verdict": e["verdict"],
        "gt": [(c, tuple(b)) for c, b in train],
        "other_gt": [(n, tuple(b)) for n, b in other],
    }


def _load_op(path: Path, schema: str) -> dict:
    op = json.loads(path.read_text(encoding="utf-8"))
    if op["schema"] != schema:
        raise ValueError(f"operating point is for schema {op['schema']!r}, not {schema!r}")
    return op


def _bundle_meta(model_path: Path) -> dict:
    meta = model_path.parent / "meta.json"
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}


def run(
    model_path: Path,
    adapter_name: str,
    schema: str,
    bucket: str,
    operating_point: Path | None = None,
    run_name: str | None = None,
    conf_floor: float = 0.001,
) -> dict:
    classes = schema_classes(schema)
    op = _load_op(operating_point, schema) if operating_point else None
    if bucket in FROZEN_OP_BUCKETS and op is None:
        raise RuntimeError(
            f"bucket {bucket!r} needs --operating-point: its threshold must be frozen "
            "on out-of-fold dev predictions (`python cli.py choose-op`), never tuned here"
        )
    if bucket == "test":
        logger.warning(
            "TEST bucket: the one-shot honest number for the FINAL model. Do not use "
            "it for model selection or threshold tuning."
        )
    if bucket == "train_always":
        logger.warning("train_always photos are always trained on: memorisation check only")

    manifest, split = load_manifest(), load_split()
    entries, probe = bucket_photos(bucket, schema)
    run_name = run_name or (
        f"{model_path.stem}_{schema}_{bucket}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    )
    out_dir = settings.eval_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = TiledDetector(model_path, adapter_name, conf_floor=conf_floor)
    n_model = getattr(detector.adapter, "num_classes", None)
    if n_model is not None and n_model != len(classes):
        raise ValueError(
            f"model outputs {n_model} classes but schema {schema!r} has {len(classes)}"
        )

    photos, total_ms = [], 0.0
    for n, e in enumerate(entries, 1):
        det = detector.detect_file(photo_path(e))
        total_ms += det.elapsed_ms
        photos.append({
            **_gt_record(e, schema, split),
            "boxes": det.boxes, "scores": det.scores, "labels": det.labels,
        })
        if n % 10 == 0 or n == len(entries):
            logger.info(f"evaluated {n}/{len(entries)} photos")

    metrics = score_all(
        photos, classes,
        threshold=op["threshold"] if op else None,
        threshold_source=f"frozen: {operating_point}" if op else "",
        probe=probe,
    )
    metrics.update({
        "run_name": run_name, "model": str(model_path), "adapter": adapter_name,
        "schema": schema, "bucket": bucket, "n_photos": len(photos),
        "mean_ms_per_photo": round(total_ms / len(photos), 1),
        "tiler_version": detector.tiler_version, "split_id": split["split_id"],
        "snapshot_id": manifest["snapshot_id"],
        "dataset_fingerprint": _bundle_meta(model_path).get("dataset_fingerprint"),
        "host": platform.node(),
    })
    _write_outputs(out_dir, metrics, photos)
    return metrics


def _write_outputs(out_dir: Path, metrics: dict, photos: list[dict]) -> None:
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "predictions.json").write_text(
        json.dumps({
            "schema": metrics["schema"], "class_names": metrics["class_names"],
            "bucket": metrics["bucket"], "model": metrics["model"],
            "photos": [
                {**{k: p[k] for k in ("capture_id", "group", "board", "pose", "gt_verdict")},
                 "gt": [[c, [round(v, 1) for v in b]] for c, b in p["gt"]],
                 "other_gt": [[n, [round(v, 1) for v in b]] for n, b in p["other_gt"]],
                 "boxes": [[round(v, 1) for v in b] for b in p["boxes"]],
                 "scores": [round(s, 5) for s in p["scores"]],
                 "labels": list(p["labels"])}
                for p in photos
            ],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    report = format_report(metrics)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    _gallery(out_dir, metrics, photos)
    _append_results_row(metrics)
    logger.info("\n" + report)
    logger.info(f"artifacts: {out_dir}")


def load_predictions(run_name: str) -> dict:
    path = settings.eval_dir / run_name / "predictions.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    for p in doc["photos"]:
        p["gt"] = [(c, tuple(b)) for c, b in p["gt"]]
        p["other_gt"] = [(n, tuple(b)) for n, b in p["other_gt"]]
        p["boxes"] = [tuple(b) for b in p["boxes"]]
    return doc


# ----------------------------------------------------------- operating point


def choose_operating_point(
    runs: list[str], target_sensitivity: float = 0.95, out: Path | None = None
) -> Path:
    """Pool out-of-fold predictions of the fold runs and freeze t*."""
    docs = [load_predictions(r) for r in runs]
    schemas = {d["schema"] for d in docs}
    buckets = [d["bucket"] for d in docs]
    if len(schemas) != 1:
        raise ValueError(f"runs mix schemas {schemas}")
    if not all(b.startswith("fold") for b in buckets) or len(set(buckets)) != len(buckets):
        raise ValueError(f"choose-op needs one run per distinct fold, got {buckets}")
    schema = schemas.pop()
    photos = [p for d in docs for p in d["photos"]]
    t = threshold_for_sensitivity(photos, target_sensitivity)
    if t is None:
        raise RuntimeError("no positive photos with a detection — cannot set t*")
    split = load_split()
    metrics = score_all(
        photos, docs[0]["class_names"], t,
        threshold_source=f"pooled out-of-fold {buckets} (t* fitted here → mildly optimistic)",
        probe=[p for p in split["probe_boxes"] if p["class"] in docs[0]["class_names"]],
    )
    op = {
        "schema": schema,
        "class_names": docs[0]["class_names"],
        "threshold": t,
        "rule": f"highest t with pooled out-of-fold photo sensitivity >= {target_sensitivity}",
        "runs": runs,
        "models": [d["model"] for d in docs],
        "split_id": split["split_id"],
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "oof_level1": metrics["level1_verdict"],
    }
    out = out or settings.eval_dir / f"operating_point_{schema}_{runs[0]}.json"
    out.write_text(json.dumps(op, indent=1, ensure_ascii=False), encoding="utf-8")
    pooled = {**metrics, "run_name": f"OOF[{', '.join(runs)}]", "schema": schema,
              "bucket": "oof_dev", "n_photos": len(photos), "model": "(pooled fold models)"}
    report = format_report(pooled)
    (out.with_suffix(".md")).write_text(report, encoding="utf-8")
    logger.info("\n" + report)
    logger.info(f"operating point frozen: t*={t:.4f} -> {out}")
    return out


# ----------------------------------------------------------------- outputs


def _fmt_rate(r: dict | None) -> str:
    if not r or r.get("rate") is None:
        return "–"
    ci = r["ci95"]
    return f"{r['rate']:.3f} ({r['k']}/{r['n']}; CI {ci[0]:.2f}–{ci[1]:.2f})"


def format_report(m: dict) -> str:
    names = m["class_names"]
    l1, l3 = m["level1_verdict"], m["level3_defects"]
    t = m["threshold"]
    lines = [
        f"# Eval {m['run_name']}",
        "",
        f"- model `{Path(str(m['model'])).name}` · schema `{m['schema']}` ({', '.join(names)}) · "
        f"bucket `{m['bucket']}` · {m['n_photos']} photos",
        f"- threshold t = {t if t is None else round(t, 4)} — {m['threshold_source']}",
    ]
    if m["bucket"] in NOT_HONEST_BUCKETS:
        lines.append(f"- ⚠ {NOT_HONEST_BUCKETS[m['bucket']]}")
    lines += [
        "",
        "## 1 · Verdict (i.O / N.iO) — acceptance level",
        "",
        f"- positives {l1['n_positive']} · negatives (iO) {l1['n_negative']} · "
        f"other-class-only (not scored) {l1['n_other_only']}",
        f"- photo AUROC {l1['photo_auroc']} (seated only: {l1['photo_auroc_seated']})",
        f"- **escape rate** {_fmt_rate(l1.get('escape'))}",
        f"- **false-call rate, seated iO** {_fmt_rate(l1.get('false_call_seated'))}",
        f"- false-call rate, handheld iO {_fmt_rate(l1.get('false_call_handheld'))}",
        f"- share of flagged photos that are iO {_fmt_rate(l1.get('false_call_share_of_flagged'))}",
    ]
    if l1.get("by_board_pose"):
        lines += ["", "| board/pose | escape | false call |", "|---|---|---|"]
        for k, v in l1["by_board_pose"].items():
            lines.append(f"| {k} | {_fmt_rate(v['escape'])} | {_fmt_rate(v['false_call'])} |")
    if "B" in {k.split('/')[0] for k in l1.get("by_board_pose", {})}:
        lines.append("\n_Board B rows = within-panel (memorisation) — board B is always trained._")

    l2 = m.get("level2_classes")
    if l2:
        lines += ["", "## 2 · Which error — class level", "",
                  "| class | photo sensitivity | class false alarm | reported on iO photos, by board |",
                  "|---|---|---|---|"]
        for name, v in l2["per_class"].items():
            by_board = ", ".join(
                f"{b}: {r['k']}/{r['n']}" for b, r in v["false_alarm_on_iO_by_board"].items()
            ) or "–"
            lines.append(f"| {SHORT.get(name, name)} | {_fmt_rate(v['photo_sensitivity'])} | "
                         f"{_fmt_rate(v['class_false_alarm'])} | {by_board} |")
        lines += ["", f"- exact class-set accuracy on N.iO photos {_fmt_rate(l2['exact_set_accuracy'])}",
                  "", "Photo-level class confusion (rows = true class present; cols = classes reported):", ""]
        cols = names + ["(none)"]
        lines += ["| true \\ reported | " + " | ".join(SHORT.get(c, c) for c in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for row, counts in l2["photo_class_confusion"].items():
            lines.append(f"| {SHORT.get(row, row)} | " + " | ".join(str(counts.get(c, 0)) for c in cols) + " |")

    lines += ["", f"## 3 · Defect level — model selection (recall @ {FPPI_OPERATING} FPPI, IoU 0.3)", "",
              "| class | GT boxes | recall@0.5FPPI (CI, unit) | AP50 | FP on other-class defects |",
              "|---|---|---|---|---|"]
    for name, v in list(l3["per_class"].items()) + [("any class", l3["class_agnostic"])]:
        if not v.get("n_gt"):
            lines.append(f"| {SHORT.get(name, name)} | 0 | – | – | – |")
            continue
        lines.append(
            f"| {SHORT.get(name, name)} | {v['n_gt']} | {v['recall_at_op']:.3f} "
            f"({v['recall_ci95'][0]:.2f}–{v['recall_ci95'][1]:.2f}, {v['ci_unit']}) | "
            f"{v['ap50']:.3f} | {v['fp_at_op_other_defect']} |"
        )
    bc = l3.get("box_confusion")
    if bc:
        lines += ["", f"Box-level confusion at t (rows = GT, cols = predicted):", "",
                  "| GT \\ pred | " + " | ".join(SHORT.get(c, c) for c in bc["columns"]) + " |",
                  "|---|" + "---|" * len(bc["columns"])]
        for row, counts in bc["rows"].items():
            lines.append(f"| {SHORT.get(row, row)} | " + " | ".join(str(counts.get(c, 0)) for c in bc["columns"]) + " |")

    probe = m.get("probe_xprod")
    if probe and probe["boxes"]:
        lines += ["", "## Cross-product probe (board-A/A2 Brücke/Fahne, never trained)", ""]
        for cls, s in probe["summary"].items():
            lines.append(f"- **{cls}**: found {s['found']}/{s['n']}, correct class {s['class_correct']}/{s['n']}")
        lines += ["", "_Counts, not rates — the only honest cross-product evidence available._"]
    if "mean_ms_per_photo" in m:
        lines += ["", f"- mean {m['mean_ms_per_photo']} ms/photo on {m.get('host')}"]
    return "\n".join(lines) + "\n"


def _gallery(out_dir: Path, m: dict, photos: list[dict]) -> None:
    t = m["threshold"]
    names = m["class_names"]
    manifest = {e["capture_id"]: e for e in load_manifest()["entries"]}
    fn_items, fp_items, conf_items = [], [], []
    for p in photos:
        keep = [i for i, s in enumerate(p["scores"]) if t is not None and s >= t]
        boxes = [p["boxes"][i] for i in keep]
        scores = [p["scores"][i] for i in keep]
        labels = [p["labels"][i] for i in keep]
        g2p, matched = match_indices(boxes, scores, [b for _, b in p["gt"]], IOU_OPERATING)
        for (c, b), j in zip(p["gt"], g2p):
            if j is None:
                fn_items.append({"capture_id": p["capture_id"], "box": b,
                                 "label": f"missed {SHORT.get(names[c], names[c])}"})
            elif labels[j] != c:
                conf_items.append({"capture_id": p["capture_id"], "box": b,
                                   "label": f"{SHORT.get(names[c], names[c])} → "
                                            f"{SHORT.get(names[labels[j]], names[labels[j]])} "
                                            f"{scores[j]:.2f}"})
        for j, ok in enumerate(matched):
            if not ok:
                other = any(iou(boxes[j], b) >= 0.1 for _, b in p["other_gt"])
                fp_items.append({"capture_id": p["capture_id"], "box": boxes[j], "score": scores[j],
                                 "label": f"FP {SHORT.get(names[labels[j]], names[labels[j]])} "
                                          f"{scores[j]:.2f}" + (" · on other defect" if other else "")})
    fp_items.sort(key=lambda x: -x["score"])
    write_gallery(
        out_dir / "gallery.html",
        {cid: photo_path(manifest[cid]) for cid in {p["capture_id"] for p in photos}},
        {"Missed defects": fn_items, "Wrong class": conf_items, "False positives": fp_items},
        f"{m['run_name']} — errors at t={t if t is None else round(t, 3)}",
    )


def _append_results_row(m: dict) -> None:
    l1, l3 = m["level1_verdict"], m["level3_defects"]
    names = m["class_names"]

    def rec(name):
        v = l3["per_class"].get(name, {})
        return v.get("recall_at_op") if v.get("n_gt") else None

    nal = l3["per_class"].get(names[0], {})
    esc = l1.get("escape") or {}
    fcs = l1.get("false_call_seated") or {}
    probe = (m.get("probe_xprod") or {}).get("summary", {})
    row = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "run_name": m["run_name"], "model": Path(str(m["model"])).name,
        "adapter": m["adapter"], "schema": m["schema"], "bucket": m["bucket"],
        "threshold": m["threshold"], "threshold_source": m["threshold_source"][:60],
        "n_photos": m["n_photos"], "n_positive": l1["n_positive"],
        "n_negative_seated": (fcs or {}).get("n"),
        "photo_auroc": l1["photo_auroc"],
        "escape": esc.get("rate"), "escape_ci_lo": (esc.get("ci95") or [None, None])[0],
        "escape_ci_hi": (esc.get("ci95") or [None, None])[1],
        "fcr_seated": fcs.get("rate"), "fcr_seated_ci_lo": (fcs.get("ci95") or [None, None])[0],
        "fcr_seated_ci_hi": (fcs.get("ci95") or [None, None])[1],
        "fcr_handheld": (l1.get("false_call_handheld") or {}).get("rate"),
        "recall_nal_0.5fppi": rec("Nicht ausreichend Lot"),
        "recall_nal_ci_lo": (nal.get("recall_ci95") or [None, None])[0],
        "recall_nal_ci_hi": (nal.get("recall_ci95") or [None, None])[1],
        "recall_bruecke_0.5fppi": rec("Brücke"), "recall_fahne_0.5fppi": rec("Fahne"),
        "recall_any_0.5fppi": l3["class_agnostic"].get("recall_at_op"),
        "exact_set_acc": ((m.get("level2_classes") or {}).get("exact_set_accuracy") or {}).get("rate"),
        "ap50_nal": nal.get("ap50"),
        "probe_found": "; ".join(f"{c} {s['found']}/{s['n']}" for c, s in probe.items()) or None,
        "mean_ms_per_photo": m["mean_ms_per_photo"], "tiler_version": m["tiler_version"],
        "split_id": m["split_id"], "snapshot_id": m["snapshot_id"],
        "dataset_fingerprint": (m.get("dataset_fingerprint") or "")[:16] or None,
        "host": m["host"],
    }
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    path = settings.reports_dir / RESULTS_FILE
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
