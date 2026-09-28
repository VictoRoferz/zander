"""
Crop verifier — the torch-free inference half of the two-stage cascade
(docs/CONTRACTS.md §8.4). numpy + onnxruntime + Pillow only: importing this
module must never load torch (the inspector image ships ml/infer without the
training stack; tests/test_verifier_infer.py checks sys.modules).

Plug-in point (offline prototype today, not wired into the inspector):

    det = TiledDetector(model, conf_floor=0.001).detect(rgb)
    det = Verifier(".../stacker.json").rescore(rgb, det)
    photo_verdict(det.boxes, det.scores, det.labels, threshold=t_star, ...)

What `rescore` does per photo:
  1. keeps the `top_k` candidates by detector score (the rest are dropped);
  2. for candidates of STACKED classes: cuts a 160-px window centred on the
     box from the full-resolution photo (ONE geometry function for training
     and inference: `extract_window`), feeds the centre 64 / 128 crops to
     verifier.onnx, and fuses [det_logit, ver_logit, log_area, seam_margin]
     with the class's logistic stacker into p;
  3. maps p onto the detector's score scale with the anchored piecewise-linear
     map A (A(tau) = t*), so infer/verdict.py keeps ONE threshold t*:
     A(p) >= t*  <=>  p >= tau  (inclusive, exactly — see `anchor_map`);
  4. high-confidence detections keep at least their detector score
     (`hi_passthrough`); PASSTHROUGH classes keep the detector score unchanged;
  5. returns Detections sorted by the final score, highest first.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image

from infer.pipeline import Detections
from infer.tiling import TILE_SIZE, tile_origins

# ---- crop geometry (CONTRACTS §8.4; changing any constant = new contract) ----
WINDOW = 160
SMALL = 64
LARGE = 128
SCALE_REF = 48
JITTER = 8
LARGE_OFFSET = (WINDOW - LARGE) // 2  # 16
SMALL_OFFSET = (WINDOW - SMALL) // 2  # 48

# seam_margin is capped at, and normalised by, this many pixels
SEAM_REF = 128

FEATURES = ("det_logit", "ver_logit", "log_area", "seam_margin")
CONTRACT_VERSION = 1
KIND = "zander-stacker"
ANCHOR = "piecewise_linear"
EXIT_DECISIONS = ("ship", "no-ship", "invalid-smoke")
LOGIT_EPS = 1e-6
INPUT_SMALL = "crop_small"
INPUT_LARGE = "crop_large"
OUTPUT_LOGITS = "logits"
OUTPUT_EMBEDDING = "embedding"
EMBED_DIM = 128

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class StackerError(ValueError):
    """stacker.json is malformed, or does not match its verifier.onnx."""


# ------------------------------------------------------------------ geometry


def window_scale(box: Sequence[float]) -> float:
    """s = max(1, max(w, h) / SCALE_REF): 1 = native pixels (every NAL box
    up to 48 px), > 1 = the window is cut larger and resized down to 160."""
    w, h = float(box[2]) - float(box[0]), float(box[3]) - float(box[1])
    return max(1.0, max(w, h) / SCALE_REF)


def extract_window(photo_rgb: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """The (160, 160, 3) uint8 window centred on `box` (photo-pixel xyxy).

    A square of side round(160 * s) centred on the box is cut from the
    full-resolution photo; only the part hanging over the photo border is
    reflect-padded (numpy "reflect" = mirror without repeating the edge row);
    the square is resized to 160 with PIL bilinear only when it is not
    already 160 (s > 1). The box centre is clamped into the photo first, so a
    window always overlaps the photo by at least half its side.
    """
    if photo_rgb.ndim != 3 or photo_rgb.shape[2] != 3 or photo_rgb.dtype != np.uint8:
        raise ValueError(
            f"photo must be an HxWx3 uint8 RGB array, got {photo_rgb.dtype} {photo_rgb.shape}"
        )
    height, width = photo_rgb.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    s = window_scale((x1, y1, x2, y2))
    side = max(WINDOW, int(math.floor(WINDOW * s + 0.5)))
    cx = min(max((x1 + x2) / 2.0, 0.0), float(width))
    cy = min(max((y1 + y2) / 2.0, 0.0), float(height))
    x0 = int(math.floor(cx - side / 2.0 + 0.5))
    y0 = int(math.floor(cy - side / 2.0 + 0.5))
    ax0, ay0 = max(0, x0), max(0, y0)
    ax1, ay1 = min(width, x0 + side), min(height, y0 + side)
    region = photo_rgb[ay0:ay1, ax0:ax1]
    (pt, pb), (pl, pr) = (ay0 - y0, y0 + side - ay1), (ax0 - x0, x0 + side - ax1)
    if side == WINDOW:
        if pt or pb or pl or pr:
            region = np.pad(region, ((pt, pb), (pl, pr), (0, 0)), mode="reflect")
    else:
        # Same pixels as reflect-padding the whole square and resizing it once,
        # but cheaper for large windows: PIL resamples horizontally first, row by
        # row, so the horizontal pass runs on the in-bounds rows only and the
        # row overhang is reflected afterwards (bit-identical, see
        # tests/test_verifier_infer.py).
        if pl or pr:
            region = np.pad(region, ((0, 0), (pl, pr), (0, 0)), mode="reflect")
        img = Image.fromarray(np.ascontiguousarray(region))
        if pt or pb:
            rows = np.asarray(img.resize((WINDOW, img.height), Image.Resampling.BILINEAR))
            img = Image.fromarray(np.pad(rows, ((pt, pb), (0, 0), (0, 0)), mode="reflect"))
        region = np.asarray(img.resize((WINDOW, WINDOW), Image.Resampling.BILINEAR))
    out = np.ascontiguousarray(region, dtype=np.uint8)
    if out.shape != (WINDOW, WINDOW, 3):  # pragma: no cover - geometry invariant
        raise AssertionError(f"window shape {out.shape}")
    return out


def crops_from_window(
    win: np.ndarray, dx: int = 0, dy: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """(small 64x64, large 128x128) HWC uint8 views, both shifted by (dx, dy).

    Training draws (dx, dy) in [-JITTER, JITTER]^2 (the same shift for both
    crops); inference uses (0, 0): large = win[16:144, 16:144],
    small = win[48:112, 48:112].
    """
    if abs(dx) > JITTER or abs(dy) > JITTER:
        raise ValueError(f"shift ({dx}, {dy}) outside +-{JITTER}")
    lo, so = LARGE_OFFSET, SMALL_OFFSET
    large = win[lo + dy: lo + dy + LARGE, lo + dx: lo + dx + LARGE]
    small = win[so + dy: so + dy + SMALL, so + dx: so + dx + SMALL]
    return small, large


def batch_inputs(windows: np.ndarray) -> dict[str, np.ndarray]:
    """ONNX feed for a [N,160,160,3] uint8 window batch: centre crops, float32
    NCHW in 0..1 (ImageNet normalisation lives inside the graph)."""
    if windows.ndim != 4 or windows.shape[1:] != (WINDOW, WINDOW, 3):
        raise ValueError(f"windows must be [N,{WINDOW},{WINDOW},3], got {windows.shape}")
    lo, so = LARGE_OFFSET, SMALL_OFFSET
    large = windows[:, lo:lo + LARGE, lo:lo + LARGE]
    small = windows[:, so:so + SMALL, so:so + SMALL]

    def nchw(a: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(a.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0

    return {INPUT_SMALL: nchw(small), INPUT_LARGE: nchw(large)}


def seam_margin(box: Sequence[float], width: int, height: int) -> float:
    """How far the box sits from the nearest tile seam, in [0, 1].

    min(128, max over tiles fully containing the box of the distance to that
    tile's INTERIOR edges) / 128. Tile edges lying on the photo border are not
    seams and do not count (a tile with no interior edge scores the cap). 0 if
    no tile contains the box. The box is clipped to the photo first. Uses the
    pinned grid of infer.tiling (the detector's own tiles).
    """
    x1 = min(max(float(box[0]), 0.0), float(width))
    y1 = min(max(float(box[1]), 0.0), float(height))
    x2 = min(max(float(box[2]), 0.0), float(width))
    y2 = min(max(float(box[3]), 0.0), float(height))
    best: Optional[float] = None
    for ox, oy in tile_origins(width, height):
        ex, ey = min(ox + TILE_SIZE, width), min(oy + TILE_SIZE, height)
        if not (x1 >= ox and y1 >= oy and x2 <= ex and y2 <= ey):
            continue
        dists = []
        if ox > 0:
            dists.append(x1 - ox)
        if ex < width:
            dists.append(ex - x2)
        if oy > 0:
            dists.append(y1 - oy)
        if ey < height:
            dists.append(ey - y2)
        margin = min(dists) if dists else math.inf
        best = margin if best is None else max(best, margin)
    if best is None:
        return 0.0
    return min(float(SEAM_REF), best) / SEAM_REF


# ------------------------------------------------------------------ scores


def logit(p, eps: float = LOGIT_EPS):
    """logit(clip(p, eps, 1 - eps)) — scalar in, float out; array in, array out."""
    a = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    out = np.log(a) - np.log1p(-a)
    return float(out) if out.ndim == 0 else out


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                   np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))
    return float(out) if out.ndim == 0 else out


def anchor_map(p, tau: float, t_star: float):
    """Anchored piecewise-linear map of a stacker probability onto the
    detector's score scale: A(p) = t*·p/tau for p < tau and
    t* + (1 - t*)(p - tau)/(1 - tau) for p >= tau.

    Monotone, A(0) = 0, A(tau) = t*, A(1) = 1, and exactly
    A(p) >= t*  <=>  p >= tau  (the p < tau branch is capped one ulp below t*,
    so floating-point rounding can never flip a verdict at the boundary).
    """
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    if not (0.0 < t_star < 1.0):
        raise ValueError(f"t* must be in (0, 1), got {t_star}")
    a = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    below = np.minimum(t_star * a / tau, np.nextafter(t_star, 0.0))
    above = t_star + (1.0 - t_star) * (a - tau) / (1.0 - tau)
    out = np.where(a >= tau, above, below)
    return float(out) if out.ndim == 0 else out


def candidate_features(
    det_scores: Sequence[float],
    ver_logits: Sequence[float],
    boxes: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> np.ndarray:
    """[N, 4] raw stacker features in FEATURES order."""
    n = len(det_scores)
    x = np.zeros((n, len(FEATURES)), dtype=np.float64)
    if n == 0:
        return x
    b = np.asarray(boxes, dtype=np.float64).reshape(n, 4)
    x[:, 0] = logit(np.asarray(det_scores, dtype=np.float64))
    x[:, 1] = np.asarray(ver_logits, dtype=np.float64)
    area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    x[:, 2] = np.log(np.maximum(area, 1.0))
    x[:, 3] = [seam_margin(bb, width, height) for bb in b]
    return x


def stack_proba(params: dict, x: np.ndarray) -> np.ndarray:
    """p = sigmoid(intercept + coef · (x - mean) / std) for one stacked class."""
    mean = np.asarray(params["mean"], dtype=np.float64)
    std = np.asarray(params["std"], dtype=np.float64)
    coef = np.asarray(params["coef"], dtype=np.float64)
    z = float(params["intercept"]) + ((np.asarray(x, dtype=np.float64) - mean) / std) @ coef
    return np.asarray(sigmoid(z), dtype=np.float64).reshape(-1)


def final_scores(params: dict, p, det_scores, t_star: float) -> np.ndarray:
    """Anchored final scores of one stacked class: A(p); where the detector
    score reaches hi_passthrough h, max(A(p), s_det)."""
    out = np.asarray(anchor_map(np.asarray(p, dtype=np.float64), params["tau"], t_star),
                     dtype=np.float64).reshape(-1)
    h = params.get("hi_passthrough")
    if h is not None:
        s = np.asarray(det_scores, dtype=np.float64).reshape(-1)
        out = np.where(s >= h, np.maximum(out, s), out)
    return out


# ------------------------------------------------------------------ stacker.json


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_stacker(doc: Any) -> None:
    """Raise StackerError listing every problem (none -> returns)."""
    errs: list[str] = []
    if not isinstance(doc, dict):
        raise StackerError("stacker.json must be a JSON object")
    if doc.get("contract_version") != CONTRACT_VERSION:
        errs.append(f"contract_version must be {CONTRACT_VERSION}")
    if doc.get("kind") != KIND:
        errs.append(f"kind must be {KIND!r}")
    names = doc.get("class_names")
    if (not isinstance(names, list) or not names
            or not all(isinstance(n, str) and n for n in names)
            or len(set(names)) != len(names)):
        errs.append("class_names must be a non-empty list of unique strings")
        names = []
    vf = doc.get("verifier_file")
    if not isinstance(vf, str) or not vf or Path(vf).name != vf:
        errs.append("verifier_file must be a bare file name")
    if not isinstance(doc.get("verifier_sha256"), str) or not _SHA256_RE.match(doc["verifier_sha256"]):
        errs.append("verifier_sha256 must be 64 lowercase hex characters")
    top_k = doc.get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        errs.append("top_k must be a positive integer")
    for key in ("candidate_floor", "detector_threshold"):
        v = doc.get(key)
        if not _num(v) or not (0.0 < v < 1.0):
            errs.append(f"{key} must be in (0, 1)")
    crop = doc.get("crop")
    want = {"window": WINDOW, "small": SMALL, "large": LARGE, "scale_ref": SCALE_REF}
    if not isinstance(crop, dict) or any(crop.get(k) != v for k, v in want.items()):
        errs.append(f"crop geometry must be {want} (this code implements only that)")
    if list(doc.get("features") or []) != list(FEATURES):
        errs.append(f"features must be {list(FEATURES)}")
    if doc.get("anchor") != ANCHOR:
        errs.append(f"anchor must be {ANCHOR!r}")
    if doc.get("knn") is not None:
        errs.append("knn must be null (the normality feature is not implemented)")
    classes = doc.get("classes")
    if not isinstance(classes, dict):
        errs.append("classes must be an object keyed by class name")
        classes = {}
    for name in names:
        c = classes.get(name)
        if not isinstance(c, dict) or c.get("mode") not in ("stack", "passthrough"):
            errs.append(f"classes[{name!r}] must have mode stack|passthrough")
            continue
        if c["mode"] == "passthrough":
            continue
        for key in ("mean", "std", "coef"):
            v = c.get(key)
            if not isinstance(v, list) or len(v) != len(FEATURES) or not all(_num(x) for x in v):
                errs.append(f"classes[{name!r}].{key} must be {len(FEATURES)} finite numbers")
        if isinstance(c.get("std"), list) and any(_num(x) and x <= 0 for x in c["std"]):
            errs.append(f"classes[{name!r}].std must be > 0")
        if not _num(c.get("intercept")):
            errs.append(f"classes[{name!r}].intercept must be a finite number")
        if not _num(c.get("tau")) or not (0.0 < c["tau"] < 1.0):
            errs.append(f"classes[{name!r}].tau must be in (0, 1)")
        h = c.get("hi_passthrough")
        if h is not None and (not _num(h) or not (0.0 < h <= 1.0)):
            errs.append(f"classes[{name!r}].hi_passthrough must be null or in (0, 1]")
    extra = set(classes) - set(names)
    if extra:
        errs.append(f"classes has entries for unknown class names {sorted(extra)}")
    ex = doc.get("exit")
    if ex is not None and (not isinstance(ex, dict) or ex.get("decision") not in EXIT_DECISIONS):
        errs.append(f"exit must be null or {{decision: {'|'.join(EXIT_DECISIONS)}, report}}")
    if errs:
        raise StackerError("invalid stacker.json: " + "; ".join(errs))


class Stacker:
    """Per-class logistic fusion + anchored map, built from stacker.json."""

    def __init__(self, doc: dict):
        validate_stacker(doc)
        self.doc = doc
        self.class_names: list[str] = list(doc["class_names"])
        self.t_star = float(doc["detector_threshold"])
        self.top_k = int(doc["top_k"])
        self.candidate_floor = float(doc["candidate_floor"])
        self.params: dict[int, dict] = {
            i: doc["classes"][n] for i, n in enumerate(self.class_names)
            if doc["classes"][n]["mode"] == "stack"
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "Stacker":
        return cls(doc)

    @property
    def stacked(self) -> set[int]:
        return set(self.params)

    @property
    def exit_decision(self) -> Optional[str]:
        ex = self.doc.get("exit")
        return ex.get("decision") if isinstance(ex, dict) else None

    def is_stacked(self, c: int) -> bool:
        return c in self.params

    @staticmethod
    def features(det_scores, ver_logits, boxes, width: int, height: int) -> np.ndarray:
        return candidate_features(det_scores, ver_logits, boxes, width, height)

    def proba(self, c: int, x: np.ndarray) -> np.ndarray:
        return stack_proba(self.params[c], x)

    def final(self, c: int, p, det_scores) -> np.ndarray:
        """Final scores for candidates of class c (passthrough -> det scores)."""
        if c not in self.params:
            return np.asarray(det_scores, dtype=np.float64).reshape(-1)
        return final_scores(self.params[c], p, det_scores, self.t_star)


# ------------------------------------------------------------------ runtime


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class Verifier:
    """verifier.onnx + stacker.json -> rescored Detections (see module doc)."""

    def __init__(
        self,
        stacker_path: Path,
        threads: int = 0,
        *,
        expected_class_names: Optional[Sequence[str]] = None,
        verify_sha: bool = True,
    ):
        import onnxruntime as ort

        self.stacker_path = Path(stacker_path)
        doc = json.loads(self.stacker_path.read_text(encoding="utf-8"))
        self.stacker = Stacker.from_dict(doc)
        self.class_names = self.stacker.class_names
        if expected_class_names is not None and list(expected_class_names) != self.class_names:
            raise StackerError(
                f"stacker class_names {self.class_names} != detector class_names "
                f"{list(expected_class_names)}"
            )
        self.onnx_path = self.stacker_path.parent / doc["verifier_file"]
        if not self.onnx_path.is_file():
            raise StackerError(f"verifier model not found: {self.onnx_path}")
        if verify_sha:
            got = sha256_file(self.onnx_path)
            if got != doc["verifier_sha256"]:
                raise StackerError(
                    f"{self.onnx_path.name} sha256 {got[:12]}... != stacker.json "
                    f"verifier_sha256 {doc['verifier_sha256'][:12]}..."
                )
        opts = ort.SessionOptions()
        if threads > 0:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(
            str(self.onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inputs = {i.name for i in self.sess.get_inputs()}
        outputs = [o.name for o in self.sess.get_outputs()]
        if inputs != {INPUT_SMALL, INPUT_LARGE} or OUTPUT_LOGITS not in outputs:
            raise StackerError(
                f"verifier.onnx must take {INPUT_SMALL}/{INPUT_LARGE} and output "
                f"{OUTPUT_LOGITS}; got inputs {sorted(inputs)} outputs {outputs}"
            )
        shape = next(o.shape for o in self.sess.get_outputs() if o.name == OUTPUT_LOGITS)
        n_out = shape[1] if len(shape) == 2 else None
        if isinstance(n_out, int) and n_out != len(self.class_names):
            raise StackerError(
                f"verifier.onnx has {n_out} class heads but stacker.json lists "
                f"{len(self.class_names)} classes"
            )

    # -- building blocks (also used by the latency bench) -------------------

    def windows(self, photo_rgb: np.ndarray, boxes: Sequence[Sequence[float]]) -> np.ndarray:
        if not len(boxes):
            return np.zeros((0, WINDOW, WINDOW, 3), dtype=np.uint8)
        return np.stack([extract_window(photo_rgb, b) for b in boxes])

    def logits(self, windows: np.ndarray) -> np.ndarray:
        """[N, C] verifier logits for a [N,160,160,3] window batch."""
        if len(windows) == 0:
            return np.zeros((0, len(self.class_names)), dtype=np.float32)
        out = self.sess.run([OUTPUT_LOGITS], batch_inputs(windows))[0]
        if out.shape != (len(windows), len(self.class_names)):
            raise StackerError(f"verifier returned logits of shape {out.shape}")
        return out

    def score_candidates(
        self,
        photo_rgb: np.ndarray,
        boxes: Sequence[Sequence[float]],
        scores: Sequence[float],
        labels: Sequence[int],
    ) -> dict:
        """Per-candidate details for the top_k candidates (by detector score):
        order (indices into the input), det, ver_logit / p (NaN for passthrough
        classes), final."""
        height, width = photo_rgb.shape[:2]
        s_all = np.asarray(scores, dtype=np.float64).reshape(-1)
        order = np.argsort(-s_all, kind="stable")[: self.stacker.top_k]
        det = s_all[order]
        lab = np.asarray(labels, dtype=np.int64).reshape(-1)[order]
        bx = [tuple(float(v) for v in boxes[i]) for i in order]
        ver = np.full(len(order), np.nan)
        p = np.full(len(order), np.nan)
        final = det.copy()
        stacked = [i for i, c in enumerate(lab) if self.stacker.is_stacked(int(c))]
        if stacked:
            lg = self.logits(self.windows(photo_rgb, [bx[i] for i in stacked]))
            for c in sorted({int(lab[i]) for i in stacked}):
                idx = [k for k, i in enumerate(stacked) if int(lab[i]) == c]
                rows = [stacked[k] for k in idx]
                ver[rows] = lg[idx, c]
                x = candidate_features(det[rows], ver[rows], [bx[i] for i in rows], width, height)
                p[rows] = self.stacker.proba(c, x)
                final[rows] = self.stacker.final(c, p[rows], det[rows])
        return {"order": order, "boxes": bx, "labels": lab, "det": det,
                "ver_logit": ver, "p": p, "final": final}

    def rescore(self, photo_rgb: np.ndarray, det: Detections) -> Detections:
        """Top-k candidates with final scores, highest first. elapsed_ms is the
        detector's time plus this call's."""
        t0 = time.perf_counter()
        r = self.score_candidates(photo_rgb, det.boxes, det.scores, det.labels)
        by_score = np.argsort(-r["final"], kind="stable")
        return Detections(
            [r["boxes"][i] for i in by_score],
            [float(r["final"][i]) for i in by_score],
            [int(r["labels"][i]) for i in by_score],
            float(det.elapsed_ms) + (time.perf_counter() - t0) * 1000.0,
        )
