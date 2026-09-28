"""
Export V_final to verifier.onnx and write stacker.json (CONTRACTS §8.4).

    python verifier/cli.py export --run-name v1

- verifier.onnx: exported from a CPU copy, opset 17, dynamic batch N; inputs
  crop_small [N,3,64,64] / crop_large [N,3,128,128] (RGB 0..1, ImageNet
  normalisation inside the graph), outputs logits [N,C] / embedding [N,128].
- Parity: torch (CPU, eval) vs onnxruntime on 32 cached windows, max abs
  difference of logits and embedding <= 1e-4, else nothing is written.
- stacker.json: S_final's per-class params, tau_final, hi_passthrough, the
  anchor t* (detector_threshold), the ONNX sha256 and full provenance; `exit`
  stays null until exit_rule.py has decided. The torch-free
  infer.verifier.Verifier must load the pair (sha + class checks) before the
  export counts as done.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from infer.verifier import (
    ANCHOR, CONTRACT_VERSION, EMBED_DIM, FEATURES, INPUT_LARGE, INPUT_SMALL, JITTER, KIND,
    LARGE, OUTPUT_EMBEDDING, OUTPUT_LOGITS, SCALE_REF, SMALL, WINDOW, Verifier, batch_inputs,
    validate_stacker,
)
from verifier.common import (
    FINAL_WEIGHTS, OOF_SCORES, STACKER, VERIFIER_ONNX, read_json, sha256_file, utc_now,
    vrun_dir, write_json_atomic,
)
from verifier.mine import load_candidates

logger = logging.getLogger("ml.verifier.export")

OPSET = 17
PARITY_N = 32
PARITY_TOL = 1e-4


def _versions() -> dict:
    import onnxruntime
    import torch
    import torchvision

    return {"torch": torch.__version__, "torchvision": torchvision.__version__,
            "onnxruntime": onnxruntime.__version__}


def export_model(model, path: Path, sample_small, sample_large) -> None:
    import torch

    torch.onnx.export(
        model, (sample_small, sample_large), str(path),
        input_names=[INPUT_SMALL, INPUT_LARGE],
        output_names=[OUTPUT_LOGITS, OUTPUT_EMBEDDING],
        dynamic_axes={INPUT_SMALL: {0: "N"}, INPUT_LARGE: {0: "N"},
                      OUTPUT_LOGITS: {0: "N"}, OUTPUT_EMBEDDING: {0: "N"}},
        opset_version=OPSET, do_constant_folding=True,
    )


def parity(model, onnx_path: Path, windows: np.ndarray) -> dict:
    """max |torch - onnxruntime| over logits and embedding for a window batch."""
    import onnxruntime as ort
    import torch

    feed = batch_inputs(windows)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    lo, eo = sess.run([OUTPUT_LOGITS, OUTPUT_EMBEDDING], feed)
    with torch.no_grad():
        lt, et = model(torch.from_numpy(feed[INPUT_SMALL]), torch.from_numpy(feed[INPUT_LARGE]))
    dl = float(np.max(np.abs(lo - lt.numpy()))) if lo.size else 0.0
    de = float(np.max(np.abs(eo - et.numpy()))) if eo.size else 0.0
    return {"n": int(len(windows)), "max_abs_logits": dl, "max_abs_embedding": de,
            "tolerance": PARITY_TOL, "ok": max(dl, de) <= PARITY_TOL,
            "logits_shape": list(lo.shape), "embedding_shape": list(eo.shape)}


def export(vrun: str) -> Path:
    import torch

    from verifier.crops import load_windows
    from verifier.model import TwoCropVerifier

    d = vrun_dir(vrun)
    cand, cand_sha = load_candidates(vrun)
    oof = read_json(d / OOF_SCORES)
    if oof["candidates_sha256"] != cand_sha:
        raise RuntimeError("oof_scores.json was built from a different candidates.json — rerun crossfit")
    ckpt = torch.load(d / FINAL_WEIGHTS, map_location="cpu", weights_only=True)
    class_names = cand["class_names"]
    if ckpt["num_classes"] != len(class_names):
        raise RuntimeError("V_final class count differs from the schema")
    model = TwoCropVerifier(len(class_names), pretrained=False, embed_dim=ckpt["embed_dim"],
                            dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if ckpt["embed_dim"] != EMBED_DIM:
        raise RuntimeError(f"embedding must be {EMBED_DIM}-d (contract), got {ckpt['embed_dim']}")

    windows = load_windows(vrun, cand_sha)
    ids = np.unique(np.linspace(0, len(windows) - 1, min(PARITY_N, len(windows))).astype(int))
    sample = np.asarray(windows[ids])
    feed = batch_inputs(sample[:2])
    tmp = d / f".{VERIFIER_ONNX}.tmp"
    export_model(model, tmp, torch.from_numpy(feed[INPUT_SMALL]), torch.from_numpy(feed[INPUT_LARGE]))
    par = parity(model, tmp, sample)
    if not par["ok"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ONNX parity failed: {par}")
    onnx_path = d / VERIFIER_ONNX
    os.replace(tmp, onnx_path)
    sha = sha256_file(onnx_path)
    logger.info(f"exported {onnx_path} (sha {sha[:12]}), parity {par}")

    doc = stacker_doc(cand, cand_sha, oof, sha, par, ckpt.get("weights_source"))
    validate_stacker(doc)
    write_json_atomic(d / STACKER, doc)
    Verifier(d / STACKER, expected_class_names=class_names)  # the inference side accepts it
    logger.info(f"stacker.json written: tau={oof['final']['tau']:.5f} anchored at "
                f"t*={oof['t_star']:.5f}")
    return d / STACKER


def stacker_doc(cand: dict, cand_sha: str, oof: dict, onnx_sha: str, par: dict,
                weights_source: Optional[str]) -> dict:
    class_names = cand["class_names"]
    final = oof["final"]
    classes = {}
    for c, name in enumerate(class_names):
        if c in oof["stack_classes"]:
            p = final["stacker"][name]
            classes[name] = {"mode": "stack", "mean": p["mean"], "std": p["std"], "coef": p["coef"],
                             "intercept": p["intercept"], "tau": final["tau"],
                             "hi_passthrough": final["hi"][name]}
        else:
            classes[name] = {"mode": "passthrough"}
    cc = cand["config"]["candidates"]
    ocfg = oof["config"]
    return {
        "contract_version": CONTRACT_VERSION, "kind": KIND,
        "schema": cand["schema"], "class_names": class_names,
        "verifier_file": VERIFIER_ONNX, "verifier_sha256": onnx_sha,
        "top_k": int(cc["top_k"]), "candidate_floor": float(cc["candidate_floor"]),
        "detector_threshold": float(oof["t_star"]),
        "crop": {"window": WINDOW, "small": SMALL, "large": LARGE, "scale_ref": SCALE_REF,
                 "jitter": JITTER, "pad": "reflect", "resize": "PIL bilinear if s > 1",
                 "input": "RGB 0..1, ImageNet normalisation inside the graph"},
        "features": list(FEATURES),
        "classes": classes,
        "anchor": ANCHOR, "knn": None,
        "provenance": {
            "vrun": cand["vrun"], "family": cand["family"], "smoke": cand["smoke"],
            "oof_violations": cand["oof_violations"], "fraction": cand["fraction"],
            "runs": [{k: r.get(k) for k in ("eval_run", "bucket", "fold", "model_sha256", "bundle_run")}
                     for r in cand["runs"]],
            "candidates_sha256": cand_sha,
            "operating_point": oof["operating_point"],
            "config": {"name": ocfg["name"], "path": ocfg["path"], "sha256": ocfg["sha256"],
                       "overrides": ocfg["overrides"]},
            "training": {"final_model_folds": list(oof["plan"]["models"]["V_final"]["train"]),
                         "epochs": ocfg["train"]["epochs"], "device": oof["device"],
                         "weights_source": weights_source},
            "stacker_fit_rows": "pooled outer out-of-fold verifier scores (V_k on fold k)",
            "tau_rule": "highest tau keeping every detector-flagged positive photo flagged, pooled folds, at t*",
            "git_sha": oof.get("git_sha"), "git_dirty": oof.get("git_dirty"),
            "versions": _versions(), "parity": par, "created": utc_now(),
        },
        "exit": None,
    }
