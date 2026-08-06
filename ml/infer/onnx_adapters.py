"""
Per-model ONNX adapters: preprocessing + raw-output decoding ONLY.

Fairness contract (from the plan): adapters may differ in what their training
framework legitimately requires (normalization, output layout), but everything
AFTER decode — confidence floor, per-tile top-k, cross-tile IoS merge,
threshold sweep — is shared code in pipeline.py/stitch.py and identical for
every model. No per-model post-processing tuning.

Both adapters take an RGB uint8 tile of exactly (TILE_SIZE, TILE_SIZE) — the
tiler guarantees this, so there is no resizing/letterboxing anywhere.

A 5-image ONNX-vs-native parity check must pass once per framework before any
benchmark row is trusted (see eval/run_eval.py --help).
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort


def _session(model_path: str, threads: int = 0) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    if threads > 0:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(
        str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


class UltralyticsAdapter:
    """
    Ultralytics YOLO exported with `nms=False` (raw head output), or an
    end-to-end NMS-free export (YOLO26) whose output is already decoded rows.

    Raw layout:  (1, 4+nc, N) or (1, N, 4+nc) — boxes cxcywh in input pixels.
    E2E layout:  (1, N, 6) — [x1, y1, x2, y2, score, class].
    """

    name = "ultralytics"

    def __init__(self, model_path: str, threads: int = 0):
        self.sess = _session(model_path, threads)
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, tile_rgb: np.ndarray) -> dict:
        x = tile_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]  # NCHW
        return {self.input_name: x}

    def decode(self, outputs: list[np.ndarray], conf_floor: float):
        out = outputs[0]
        if out.ndim != 3:
            raise ValueError(f"unexpected output shape {out.shape}")
        if out.shape[2] == 6:  # end-to-end export: rows [xyxy, score, cls]
            rows = out[0]
            keep = rows[:, 4] >= conf_floor
            return rows[keep, :4].copy(), rows[keep, 4].copy()
        if out.shape[1] < out.shape[2]:  # (1, 4+nc, N) -> (N, 4+nc)
            rows = out[0].T
        else:
            rows = out[0]
        scores = rows[:, 4:].max(axis=1)
        keep = scores >= conf_floor
        rows, scores = rows[keep], scores[keep]
        cx, cy, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        return boxes, scores

    def run(self, tile_rgb: np.ndarray, conf_floor: float):
        outputs = self.sess.run(None, self.preprocess(tile_rgb))
        return self.decode(outputs, conf_floor)


class DeimAdapter:
    """
    DEIM / D-FINE deploy export: inputs (images, orig_target_sizes), outputs
    (labels, boxes xyxy in original pixels, scores). NMS-free top-300 decode
    is inside the graph. Input normalization is /255 only (RT-DETR family
    trains without ImageNet mean/std).
    """

    name = "deim"

    def __init__(self, model_path: str, threads: int = 0):
        self.sess = _session(model_path, threads)
        names = [i.name for i in self.sess.get_inputs()]
        self.image_input = names[0]
        self.size_input = names[1] if len(names) > 1 else None

    def run(self, tile_rgb: np.ndarray, conf_floor: float):
        h, w = tile_rgb.shape[:2]
        x = tile_rgb.astype(np.float32) / 255.0
        feed = {self.image_input: np.transpose(x, (2, 0, 1))[None]}
        if self.size_input:
            feed[self.size_input] = np.array([[w, h]], dtype=np.int64)
        outputs = self.sess.run(None, feed)
        # Identify (labels, boxes, scores) by shape, robust to output order.
        boxes = scores = None
        for out in outputs:
            arr = out[0] if out.ndim == 3 or out.ndim == 2 else out
            if arr.ndim == 2 and arr.shape[1] == 4:
                boxes = arr
            elif arr.ndim == 1 and arr.dtype in (np.float32, np.float64):
                scores = arr
        if boxes is None or scores is None:
            raise ValueError(
                f"could not identify DEIM outputs from shapes "
                f"{[o.shape for o in outputs]}"
            )
        keep = scores >= conf_floor
        return boxes[keep].copy(), scores[keep].copy()


ADAPTERS = {"ultralytics": UltralyticsAdapter, "deim": DeimAdapter}


def load_adapter(name: str, model_path: str, threads: int = 0):
    if name not in ADAPTERS:
        raise KeyError(f"unknown adapter {name!r} — one of {sorted(ADAPTERS)}")
    return ADAPTERS[name](model_path, threads=threads)
