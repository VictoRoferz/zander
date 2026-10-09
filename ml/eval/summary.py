"""
Model-selection summary over the pooled out-of-fold evaluations — run on the
Mac after `eval-matrix` (and ideally `choose-op` per family).

    python cli.py summary [--include-smoke]

Per family F (fold runs F_fold0..2 in ml_data/eval, own-fold, distinct
models — see eval/compare.resolve):
  - pooled OOF photos = the fold runs concatenated -> eval/photo_scorer.score_all
    at F's FROZEN operating point (operating_point_<schema>_<F>.json, which must
    have been frozen from exactly these runs' models — a stale op excludes the
    family); without one the threshold is self-tuned on the pooled OOF and
    FLAGGED (optimistic);
  - pooled NAL recall @ 0.5 FPPI with board-group bootstrap CIs, per-fold
    recalls, AP50 (literature comparison only), photo AUROC, escape and
    false-call rates with their counts, probe hits (never-trained board-A
    Brücke/Fahne boxes), ms/photo on this host, ONNX layout.
Selection (plan; decisions are paired, never read off overlapping CIs):
  - rank the def3 families (the product) by pooled NAL recall (ties: faster);
  - G1: top-1 vs top-2 paired comparison — a winner needs a CI excluding 0,
    otherwise prefer the faster/simpler model;
  - NAL gate: each def3 family must be non-inferior to its <arch>_nal1
    counterpart on NAL recall within margin 0.05.
Incomplete families (missing folds) are listed, never ranked.

Writes ml/reports/model_selection_v2.md (committed) and ml_data/eval/summary_v2.json.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import registry
from config.settings import settings
from dataprep.remap import BOX_CLASSES
from dataprep.split import load_split
from eval import compare
from eval.photo_scorer import score_all, threshold_for_sensitivity
from eval.run_eval import (
    EvalGuardError,
    _atomic_write_text,
    check_op_sources,
    load_predictions,
    run_source,
    source_bundle_info,
)

logger = logging.getLogger("ml.summary")

NAL = BOX_CLASSES[0]
NAL_GATE_MARGIN = 0.05
PRODUCT_SCHEMA = "def3"
_FOLD_RUN_RE = re.compile(r"^(?P<family>.+)_fold(?P<k>\d+)$")


def discover_families(include_smoke: bool = False) -> dict[str, dict[int, str]]:
    """family -> {fold index: eval run} for detector fold runs in ml_data/eval."""
    fams: dict[str, dict[int, str]] = {}
    for d in sorted(settings.eval_dir.iterdir()) if settings.eval_dir.is_dir() else []:
        m = _FOLD_RUN_RE.match(d.name)
        if not m or not (d / "predictions.json").is_file() or not (d / "metrics.json").is_file():
            continue
        family = m.group("family")
        if "+" in family or "__" in family:  # cascade runs / non-fold buckets
            continue
        if registry.is_smoke(family) and not include_smoke:
            continue
        fams.setdefault(family, {})[int(m.group("k"))] = d.name
    return fams


def _rate(r: dict | None) -> dict | None:
    return None if not r else {k: r.get(k) for k in ("k", "n", "rate", "ci95")}


def _family_entry(family: str, runs: dict[int, str], k: int) -> tuple[dict | None, dict | None]:
    """(summary entry, resolved pool for compare) — (None, None) + reason on refusal."""
    complete = sorted(runs) == list(range(k))
    if complete:
        res = compare.resolve(family)  # own-fold / distinct-model guards
        docs = res["docs"]
    else:
        docs = [load_predictions(runs[i]) for i in sorted(runs)]
        problems = compare.pool_problems([runs[i] for i in sorted(runs)], docs)
        if problems:
            raise EvalGuardError(f"family {family!r} cannot be pooled: " + "; ".join(problems))
        res = None
    schema, class_names = docs[0]["schema"], docs[0]["class_names"]
    photos = [p for d in docs for p in d["photos"]]
    op_path = settings.eval_dir / f"operating_point_{schema}_{family}.json"
    if op_path.is_file():
        op = json.loads(op_path.read_text(encoding="utf-8"))
        check_op_sources(op, op_path.name, docs)  # a stale op is refused, never applied
        t = op["threshold"]
        t_src = f"frozen: {op_path.name}"
    else:
        t = threshold_for_sensitivity(photos)
        t_src = "SELF-TUNED on the pooled OOF (no choose-op yet: optimistic)"
    split = load_split()
    ids = {p["capture_id"] for p in photos}
    probe = [p for p in split["probe_boxes"] if p["class"] in class_names and p["capture_id"] in ids]
    m = score_all(photos, class_names, t, t_src, probe=probe)
    l1, l3 = m["level1_verdict"], m["level3_defects"]
    nal = l3["per_class"].get(NAL) or {}
    fold_runs = [runs[i] for i in sorted(runs)]
    per_fold, ms, n_ms, layouts, shas = {}, 0.0, 0, set(), []
    for r in fold_runs:
        fm = json.loads((settings.eval_dir / r / "metrics.json").read_text(encoding="utf-8"))
        per_fold[r] = registry.eval_key_metrics(fm)["recall_nal_0.5fppi"]
        if fm.get("mean_ms_per_photo") is not None:
            ms += fm["mean_ms_per_photo"] * fm["n_photos"]
            n_ms += fm["n_photos"]
    for d in docs:
        src = run_source(d)
        shas.append(src.get("model_sha256"))
        info = source_bundle_info(src)
        layouts.add((info or {}).get("onnx_layout")
                    or ("unknown (bundle v1)" if info and info["bundle_version"] < 2 else "unknown"))
    entry = {
        "family": family, "schema": schema, "class_names": class_names,
        "arch": family[: -len(schema) - 1] if family.endswith(f"_{schema}") else family,
        "complete": complete, "folds": f"{len(runs)}/{k}", "runs": fold_runs,
        "n_photos": len(photos), "threshold": t, "threshold_source": t_src,
        "self_tuned": not op_path.is_file(),
        "nal": {"n_gt": nal.get("n_gt", 0), "recall": nal.get("recall_at_op"),
                "ci95": nal.get("recall_ci95"), "ap50": nal.get("ap50")},
        "any_class": {"recall": l3["class_agnostic"].get("recall_at_op"),
                      "ci95": l3["class_agnostic"].get("recall_ci95")},
        "per_class_recall": {n: v.get("recall_at_op") for n, v in l3["per_class"].items() if v.get("n_gt")},
        "per_fold_nal_recall": per_fold,
        "photo_auroc": l1.get("photo_auroc"),
        "escape": _rate(l1.get("escape")),
        "fcr_seated": _rate(l1.get("false_call_seated")),
        "fcr_handheld": _rate(l1.get("false_call_handheld")),
        "probe": (m.get("probe_xprod") or {}).get("summary", {}),
        "ms_per_photo": round(ms / n_ms, 1) if n_ms else None,
        "onnx_layout": ", ".join(sorted(str(x) for x in layouts)),
        "model_sha256": shas,
        "smoke": registry.is_smoke(family),
    }
    return entry, res


def _fmt_rate(r: dict | None) -> str:
    if not r or r.get("rate") is None:
        return "–"
    return f"{r['rate']:.3f} ({r['k']}/{r['n']})"


def _fmt_recall(v: dict) -> str:
    if v.get("recall") is None:
        return "–"
    ci = v.get("ci95") or [None, None]
    return f"**{v['recall']:.3f}** ({ci[0]:.2f}–{ci[1]:.2f})"


def _table(entries: list[dict], ranked: bool) -> list[str]:
    head = ("| rank | family | folds | NAL recall@0.5FPPI pooled (group CI) | per fold | AP50 NAL "
            "| photo AUROC | t (source) | escape (k/n) | false call seated (k/n) | probe found "
            "| ms/photo | ONNX |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for i, e in enumerate(entries, 1):
        per_fold = " / ".join("–" if v is None else f"{v:.2f}" for v in e["per_fold_nal_recall"].values())
        probe = "; ".join(f"{c} {s['found']}/{s['n']}" for c, s in e["probe"].items()) or "–"
        t = "–" if e["threshold"] is None else f"{e['threshold']:.4g}"
        t_src = "self-tuned ⚠" if e["self_tuned"] else "frozen"
        ap = e["nal"].get("ap50")
        lines.append(
            f"| {i if ranked else '–'} | `{e['family']}` | {e['folds']} | {_fmt_recall(e['nal'])} | "
            f"{per_fold} | {'–' if ap is None else f'{ap:.3f}'} | {e['photo_auroc']} | {t} ({t_src}) | "
            f"{_fmt_rate(e['escape'])} | {_fmt_rate(e['fcr_seated'])} | {probe} | "
            f"{e['ms_per_photo']} | {e['onnx_layout']} |"
        )
    return lines


def run(
    include_smoke: bool = False,
    families: list[str] | None = None,
    out_md: Path | None = None,
    out_json: Path | None = None,
    seed: int = 0,
) -> dict:
    split = load_split()
    k = len(split["folds"])
    found = discover_families(include_smoke=include_smoke)
    if families:
        found = {f: found[f] for f in families if f in found}
    entries, pools, excluded = [], {}, []
    for family, runs in sorted(found.items()):
        try:
            entry, res = _family_entry(family, runs, k)
        except (EvalGuardError, ValueError, FileNotFoundError) as e:
            excluded.append({"family": family, "reason": str(e)})
            logger.warning(f"summary: {family} excluded — {e}")
            continue
        entries.append(entry)
        if res is not None:
            pools[family] = res

    def rank_key(e):
        return (-(e["nal"]["recall"] or 0.0), e["ms_per_photo"] or float("inf"))

    complete = [e for e in entries if e["complete"]]
    candidates = sorted([e for e in complete if e["schema"] == PRODUCT_SCHEMA], key=rank_key)
    nal_only = sorted([e for e in complete if e["schema"] != PRODUCT_SCHEMA], key=rank_key)
    incomplete = [e for e in entries if not e["complete"]]

    def paired(a: str, b: str, **kw) -> dict:
        try:
            return compare.compare(a, b, cls="NAL", seed=seed, resolved=(pools[a], pools[b]), **kw)
        except (compare.CompareError, EvalGuardError) as e:  # e.g. different photo sets
            logger.warning(f"summary: {a} vs {b} not comparable — {e}")
            return {"run_a": a, "run_b": b, "error": str(e)}

    g1 = None
    if len(candidates) >= 2:
        g1 = paired(candidates[0]["family"], candidates[1]["family"])
    gates = []
    for e in candidates:
        twin = f"{e['arch']}_nal1"
        if twin in pools:
            gates.append(paired(e["family"], twin, margin=NAL_GATE_MARGIN))
        else:
            gates.append({"run_a": e["family"], "run_b": twin, "missing": True})

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc = {
        "created": now, "split_id": split["split_id"], "k_folds": k,
        "include_smoke": include_smoke, "nal_gate_margin": NAL_GATE_MARGIN,
        "ranking": [e["family"] for e in candidates],
        "families": entries, "excluded": excluded, "g1": g1, "nal_gate": gates,
    }
    out_json = Path(out_json) if out_json else settings.eval_dir / "summary_v2.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(out_json, json.dumps(doc, indent=1, ensure_ascii=False, default=str))

    md = [
        "# Model selection v2 — pooled out-of-fold",
        "",
        f"_Generated {now} by `python cli.py summary` from `ml_data/eval` "
        f"(`{out_json.name}`); split `{split['split_id']}`, {k} folds._",
        "",
        "Each family's fold models are scored only on their own validation fold; the "
        "three runs are pooled (concatenated) into one out-of-fold set. CIs = board-group "
        "cluster bootstrap. Thresholds come from the family's frozen operating point "
        "(`choose-op`); ⚠ self-tuned thresholds are optimistic and only a placeholder. "
        "Decisions use paired comparisons (`compare`), never overlapping CIs; AP50 is for "
        "literature comparison only.",
        "",
        f"## Candidates ({PRODUCT_SCHEMA}) — ranked by pooled NAL recall @ 0.5 FPPI",
        "",
    ]
    md += _table(candidates, ranked=True) if candidates else ["_No complete def3 family yet._"]
    md += ["", "## G1 — top-1 vs top-2 (paired, NAL)", ""]
    if g1 and g1.get("error"):
        md.append(f"- `{g1['run_a']}` vs `{g1['run_b']}`: not comparable — {g1['error']}")
    elif g1:
        md.append(f"- {compare.summary_line(g1)}")
        md.append(f"- **Decision:** {g1['g1_decision']}"
                  + (" (see ms/photo above)" if "no significant" in g1["g1_decision"] else ""))
    else:
        md.append("_Needs two complete def3 families._")
    md += ["", f"## NAL gate — def3 non-inferior to nal1 (margin {NAL_GATE_MARGIN})", ""]
    if gates:
        for g in gates:
            if g.get("missing"):
                md.append(f"- `{g['run_a']}`: no complete `{g['run_b']}` family — gate not evaluated")
            elif g.get("error"):
                md.append(f"- `{g['run_a']}` vs `{g['run_b']}`: not comparable — {g['error']}")
            else:
                md.append(f"- `{g['run_a']}` vs `{g['run_b']}`: NAL recall {g['recall_a']:.3f} vs "
                          f"{g['recall_b']:.3f}, diff {g['recall_at_0.5fppi_diff']:+.3f} (CI "
                          f"{g['recall_diff_ci95']}) → **{g['non_inferiority']}**")
    else:
        md.append("_No complete def3 family yet._")
    if nal_only:
        md += ["", "## NAL-only families (nal1, comparison only)", ""] + _table(nal_only, ranked=False)
    if incomplete:
        md += ["", "## Incomplete families (not ranked)", ""] + _table(incomplete, ranked=False)
    if excluded:
        md += ["", "## Excluded", ""] + [f"- `{x['family']}`: {x['reason']}" for x in excluded]
    if any(e["smoke"] for e in entries):
        md += ["", "_⚠ Smoke families included (`--include-smoke`): plumbing checks, not models._"]
    out_md = Path(out_md) if out_md else settings.reports_dir / "model_selection_v2.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(out_md, "\n".join(md) + "\n")
    logger.info(f"summary: {len(candidates)} ranked, {len(incomplete)} incomplete, "
                f"{len(excluded)} excluded -> {out_md}, {out_json}")
    return doc
