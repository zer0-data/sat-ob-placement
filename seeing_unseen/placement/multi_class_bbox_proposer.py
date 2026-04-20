"""
MultiClassBBoxProposer
======================
Converts a dict of per-class affordance heatmaps into a single ranked list of
class-aware placement bounding boxes.

Pipeline
--------
1. Per-class calibration (percentile / zscore / minmax) so heatmaps are
   comparable across classes.
2. Per-pixel argmax + per-class threshold → hard class assignment
   (``class_map`` with value -1 for "none").
3. Per class, connected-component labeling of the binary mask, then tile each
   blob with (box_w, box_h) windows at a configurable stride. Filter tiles by
   coverage of the class mask.
4. Score each candidate by the raw (un-normalized) heatmap inside the box
   (``mean`` / ``max`` / ``sum``).
5. Class-agnostic greedy NMS over all candidates.
6. Optional per-class and global caps.

Usage
-----
    proposer = MultiClassBBoxProposer(
        class_specs=[
            ClassSpec("S-400", 80, 40, threshold=0.35, max_boxes=5),
            ClassSpec("tank",  24, 16, threshold=0.30, max_boxes=10),
        ],
    )
    result = proposer.propose(heatmaps={"S-400": hm1, "tank": hm2})
    for b in result.boxes:
        print(b.class_name, b.x1, b.y1, b.x2, b.y2, b.score)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from scipy.ndimage import label as _cc_label
    _SCIPY_CC = True
except ImportError:
    _SCIPY_CC = False

try:
    from torchvision.ops import nms as _tv_nms
    _TV_NMS = True
except ImportError:
    _TV_NMS = False


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClassSpec:
    name: str
    box_w: int
    box_h: int
    threshold: Optional[float] = None
    max_boxes: Optional[int] = None


@dataclass
class Box:
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    def as_tuple(self) -> Tuple[int, int, int, int, float, str]:
        return (self.x1, self.y1, self.x2, self.y2, self.score, self.class_name)


@dataclass
class MultiClassResult:
    boxes: List[Box]
    class_map: np.ndarray                           # (H, W) int32, -1 for none
    normalized: Dict[str, np.ndarray] = field(default_factory=dict)
    binary_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    pre_nms_candidates: List[Box] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MultiClassBBoxProposer
# ---------------------------------------------------------------------------

class MultiClassBBoxProposer:
    """
    Parameters
    ----------
    class_specs         : ordered list of ClassSpec. Class index == position.
    default_threshold   : fallback threshold used when ClassSpec.threshold is None.
    normalization       : "percentile" | "zscore" | "minmax" | "none"
    percentile_lo,
    percentile_hi       : percentiles for ``percentile`` normalization.
    min_blob_area       : drop connected components smaller than this (pixels).
    tile_stride_frac    : stride between tiled candidate boxes, as a fraction of
                          the box dimensions. 0.5 → 50% overlap. Must be in (0, 1].
    min_coverage        : minimum fraction of pixels inside a candidate box that
                          must belong to the class mask. Relaxed to 0 when the
                          blob is smaller than the box (small-blob fallback).
    score_fn            : "mean" | "max" | "sum"
    nms_iou_thresh      : IoU threshold for class-agnostic NMS.
    max_boxes_total     : optional cap on the final output length.
    """

    def __init__(
        self,
        class_specs: List[ClassSpec],
        default_threshold: float = 0.35,
        normalization: str = "percentile",
        percentile_lo: float = 50.0,
        percentile_hi: float = 99.0,
        min_blob_area: int = 25,
        tile_stride_frac: float = 0.5,
        min_coverage: float = 0.3,
        score_fn: str = "mean",
        nms_iou_thresh: float = 0.3,
        max_boxes_total: Optional[int] = None,
    ) -> None:
        if not class_specs:
            raise ValueError("class_specs must be non-empty.")
        if normalization not in ("percentile", "zscore", "minmax", "none"):
            raise ValueError(f"Unknown normalization: {normalization!r}")
        if score_fn not in ("mean", "max", "sum"):
            raise ValueError(f"Unknown score_fn: {score_fn!r}")
        if not (0.0 < tile_stride_frac <= 1.0):
            raise ValueError("tile_stride_frac must be in (0, 1].")
        if not (0.0 <= min_coverage <= 1.0):
            raise ValueError("min_coverage must be in [0, 1].")
        if not (0.0 <= percentile_lo < percentile_hi <= 100.0):
            raise ValueError("Require 0 <= percentile_lo < percentile_hi <= 100.")

        for spec in class_specs:
            if spec.box_w <= 0 or spec.box_h <= 0:
                raise ValueError(
                    f"Class {spec.name!r}: box_w/box_h must be positive, "
                    f"got {spec.box_w}×{spec.box_h}."
                )

        self.class_specs = list(class_specs)
        self.default_threshold = float(default_threshold)
        self.normalization = normalization
        self.percentile_lo = float(percentile_lo)
        self.percentile_hi = float(percentile_hi)
        self.min_blob_area = int(min_blob_area)
        self.tile_stride_frac = float(tile_stride_frac)
        self.min_coverage = float(min_coverage)
        self.score_fn = score_fn
        self.nms_iou_thresh = float(nms_iou_thresh)
        self.max_boxes_total = (
            int(max_boxes_total) if max_boxes_total is not None else None
        )

        self._class_to_idx = {s.name: i for i, s in enumerate(class_specs)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose(
        self,
        heatmaps: Dict[str, torch.Tensor],
        validity_mask: Optional[torch.Tensor] = None,
    ) -> MultiClassResult:
        """
        Parameters
        ----------
        heatmaps      : dict {class_name: (1,H,W) or (H,W) float tensor in [0,1]}.
                        Must contain an entry for every class in class_specs.
        validity_mask : optional (1,H,W) or (H,W) binary tensor. Pixels with
                        value 0 are excluded from assignment (treated as "none").

        Returns
        -------
        MultiClassResult
        """
        raw = self._collect_heatmaps(heatmaps)           # (C, H, W) np.float32
        C, H, W = raw.shape

        # Normalization → calibrated per-class scores (C, H, W)
        calibrated = self._calibrate(raw)

        # Validity mask (H, W) 0/1
        validity = self._prepare_validity(validity_mask, H, W)

        # Hard assignment: argmax + per-class threshold + validity
        class_map = self._assign(calibrated, validity)   # (H, W) int32, -1 for none

        # Per-class proposals
        pre_nms: List[Box] = []
        binary_masks: Dict[str, np.ndarray] = {}
        for cid, spec in enumerate(self.class_specs):
            mask = (class_map == cid)
            binary_masks[spec.name] = mask
            if not mask.any():
                continue
            pre_nms.extend(
                self._propose_for_class(
                    spec=spec,
                    mask=mask,
                    raw_heatmap=raw[cid],
                    H=H,
                    W=W,
                )
            )

        # Class-agnostic NMS
        kept = self._nms(pre_nms)

        # Per-class + global caps
        final = self._apply_caps(kept)

        normalized_dict = {
            spec.name: calibrated[cid]
            for cid, spec in enumerate(self.class_specs)
        }

        return MultiClassResult(
            boxes=final,
            class_map=class_map,
            normalized=normalized_dict,
            binary_masks=binary_masks,
            pre_nms_candidates=pre_nms,
        )

    # ------------------------------------------------------------------
    # Step 0 — tensor wrangling
    # ------------------------------------------------------------------

    def _collect_heatmaps(
        self, heatmaps: Dict[str, torch.Tensor]
    ) -> np.ndarray:
        missing = [s.name for s in self.class_specs if s.name not in heatmaps]
        if missing:
            raise KeyError(f"heatmaps missing class(es): {missing}")

        stacked: List[np.ndarray] = []
        shape_ref: Optional[Tuple[int, int]] = None
        for spec in self.class_specs:
            hm = heatmaps[spec.name]
            arr = hm.squeeze().detach().cpu().float().numpy()
            if arr.ndim != 2:
                raise ValueError(
                    f"Heatmap for {spec.name!r} must be (H,W) or (1,H,W); "
                    f"got shape {tuple(hm.shape)}."
                )
            if shape_ref is None:
                shape_ref = arr.shape
            elif arr.shape != shape_ref:
                raise ValueError(
                    f"Heatmap shape mismatch: {spec.name!r} is {arr.shape}, "
                    f"expected {shape_ref}."
                )
            stacked.append(arr.astype(np.float32, copy=False))
        return np.stack(stacked, axis=0)                 # (C, H, W)

    def _prepare_validity(
        self,
        validity_mask: Optional[torch.Tensor],
        H: int,
        W: int,
    ) -> np.ndarray:
        if validity_mask is None:
            return np.ones((H, W), dtype=bool)
        arr = validity_mask.squeeze().detach().cpu().float().numpy()
        if arr.shape != (H, W):
            raise ValueError(
                f"validity_mask shape {arr.shape} != heatmap shape {(H, W)}."
            )
        return arr > 0.5

    # ------------------------------------------------------------------
    # Step 1 — Calibration
    # ------------------------------------------------------------------

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        if self.normalization == "none":
            return raw.copy()

        out = np.empty_like(raw)
        for cid in range(raw.shape[0]):
            out[cid] = self._calibrate_one(raw[cid])
        return out

    def _calibrate_one(self, hm: np.ndarray) -> np.ndarray:
        if self.normalization == "percentile":
            lo = np.percentile(hm, self.percentile_lo)
            hi = np.percentile(hm, self.percentile_hi)
            if hi - lo < 1e-8:
                return np.zeros_like(hm)
            return np.clip((hm - lo) / (hi - lo), 0.0, 1.0)

        if self.normalization == "zscore":
            mu = float(hm.mean())
            sd = float(hm.std())
            if sd < 1e-8:
                return np.zeros_like(hm)
            z = (hm - mu) / sd
            return 1.0 / (1.0 + np.exp(-z))              # logistic squash

        if self.normalization == "minmax":
            lo = float(hm.min())
            hi = float(hm.max())
            if hi - lo < 1e-8:
                return np.zeros_like(hm)
            return (hm - lo) / (hi - lo)

        # unreachable
        raise AssertionError(self.normalization)

    # ------------------------------------------------------------------
    # Step 2 — Hard assignment
    # ------------------------------------------------------------------

    def _assign(
        self, calibrated: np.ndarray, validity: np.ndarray
    ) -> np.ndarray:
        C = calibrated.shape[0]
        argmax = calibrated.argmax(axis=0).astype(np.int32)        # (H, W)
        top_score = np.take_along_axis(
            calibrated, argmax[None], axis=0
        ).squeeze(0)                                               # (H, W)

        thresholds = np.array(
            [
                s.threshold if s.threshold is not None else self.default_threshold
                for s in self.class_specs
            ],
            dtype=np.float32,
        )
        per_pixel_thresh = thresholds[argmax]                      # (H, W)

        keep = (top_score >= per_pixel_thresh) & validity
        class_map = np.where(keep, argmax, -1).astype(np.int32)
        return class_map

    # ------------------------------------------------------------------
    # Step 3 — Per-class proposals via connected components + tiling
    # ------------------------------------------------------------------

    def _propose_for_class(
        self,
        spec: ClassSpec,
        mask: np.ndarray,
        raw_heatmap: np.ndarray,
        H: int,
        W: int,
    ) -> List[Box]:
        components = self._connected_components(mask)
        if not components:
            return []

        bw, bh = spec.box_w, spec.box_h
        stride_x = max(1, int(round(bw * self.tile_stride_frac)))
        stride_y = max(1, int(round(bh * self.tile_stride_frac)))

        max_x1 = max(0, W - bw)
        max_y1 = max(0, H - bh)

        boxes: List[Box] = []
        for comp in components:
            if comp["area"] < self.min_blob_area:
                continue

            y0, x0, y1b, x1b = comp["bbox"]          # inclusive y1b, x1b
            blob_w = x1b - x0 + 1
            blob_h = y1b - y0 + 1

            small_blob = blob_w < bw or blob_h < bh

            if small_blob:
                # Single fallback box centered on centroid, clipped to image.
                cy, cx = comp["centroid"]
                x1 = int(np.clip(round(cx - bw / 2), 0, max_x1))
                y1 = int(np.clip(round(cy - bh / 2), 0, max_y1))
                x2 = x1 + bw
                y2 = y1 + bh
                score = self._score(raw_heatmap[y1:y2, x1:x2])
                boxes.append(Box(spec.name, x1, y1, x2, y2, score))
                continue

            # Slide candidate boxes across the blob's bounding rect.
            # Candidate upper-left x ranges over [x0 - (bw - blob_w_at_edge), x1b - bw + 1]
            # but we simply iterate over positions where the box fits in the image
            # and intersects the blob bbox.
            x_start = max(0, x0)
            y_start = max(0, y0)
            # Position so that the RIGHT edge of the box can reach x1b.
            # i.e. upper-left x ranges [x0 - bw + 1, x1b], clipped to [0, max_x1].
            x_lo = max(0, x0 - bw + 1)
            x_hi = min(max_x1, x1b)
            y_lo = max(0, y0 - bh + 1)
            y_hi = min(max_y1, y1b)

            # Ensure we always emit at least the anchor at (x_start, y_start)
            # even if stride would skip the bbox.
            xs = list(range(x_lo, x_hi + 1, stride_x))
            if not xs or xs[-1] != x_hi:
                xs.append(x_hi)
            ys = list(range(y_lo, y_hi + 1, stride_y))
            if not ys or ys[-1] != y_hi:
                ys.append(y_hi)

            for y1 in ys:
                for x1 in xs:
                    x2 = x1 + bw
                    y2 = y1 + bh
                    tile_mask = mask[y1:y2, x1:x2]
                    coverage = float(tile_mask.mean()) if tile_mask.size else 0.0
                    if coverage < self.min_coverage:
                        continue
                    score = self._score(raw_heatmap[y1:y2, x1:x2])
                    boxes.append(Box(spec.name, x1, y1, x2, y2, score))

        return boxes

    def _score(self, region: np.ndarray) -> float:
        if region.size == 0:
            return 0.0
        if self.score_fn == "max":
            return float(region.max())
        if self.score_fn == "sum":
            return float(region.sum())
        return float(region.mean())

    def _connected_components(self, mask: np.ndarray) -> List[dict]:
        """Return list of {area, bbox=(y0,x0,y1,x1 inclusive), centroid=(cy,cx)}."""
        if not mask.any():
            return []

        if _SCIPY_CC:
            structure = np.ones((3, 3), dtype=np.int32)   # 8-connectivity
            labels, n = _cc_label(mask, structure=structure)
        else:
            labels, n = self._cc_numpy(mask)

        out: List[dict] = []
        for lid in range(1, n + 1):
            ys, xs = np.where(labels == lid)
            if ys.size == 0:
                continue
            out.append(
                {
                    "area": int(ys.size),
                    "bbox": (int(ys.min()), int(xs.min()),
                             int(ys.max()), int(xs.max())),
                    "centroid": (float(ys.mean()), float(xs.mean())),
                }
            )
        return out

    @staticmethod
    def _cc_numpy(mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Pure-numpy 8-connected component labeling (scipy fallback)."""
        H, W = mask.shape
        labels = np.zeros((H, W), dtype=np.int32)
        next_label = 0
        # Iterative flood-fill — O(H*W) with a stack per component.
        for i in range(H):
            for j in range(W):
                if not mask[i, j] or labels[i, j] != 0:
                    continue
                next_label += 1
                stack = [(i, j)]
                while stack:
                    y, x = stack.pop()
                    if y < 0 or y >= H or x < 0 or x >= W:
                        continue
                    if not mask[y, x] or labels[y, x] != 0:
                        continue
                    labels[y, x] = next_label
                    stack.extend(
                        [
                            (y - 1, x - 1), (y - 1, x), (y - 1, x + 1),
                            (y, x - 1),                  (y, x + 1),
                            (y + 1, x - 1), (y + 1, x), (y + 1, x + 1),
                        ]
                    )
        return labels, next_label

    # ------------------------------------------------------------------
    # Step 4 — NMS
    # ------------------------------------------------------------------

    def _nms(self, candidates: List[Box]) -> List[Box]:
        if not candidates:
            return []
        boxes = np.array(
            [[b.x1, b.y1, b.x2, b.y2] for b in candidates], dtype=np.float32
        )
        scores = np.array([b.score for b in candidates], dtype=np.float32)

        if _TV_NMS:
            keep_idx = _tv_nms(
                torch.from_numpy(boxes),
                torch.from_numpy(scores),
                self.nms_iou_thresh,
            ).tolist()
        else:
            keep_idx = self._nms_numpy(boxes, scores)

        return [candidates[i] for i in keep_idx]

    def _nms_numpy(
        self, boxes: np.ndarray, scores: np.ndarray
    ) -> List[int]:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(-scores)
        keep: List[int] = []
        while len(order):
            i = int(order[0])
            keep.append(i)
            if len(order) == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
            order = rest[iou <= self.nms_iou_thresh]
        return keep

    # ------------------------------------------------------------------
    # Step 5 — Caps
    # ------------------------------------------------------------------

    def _apply_caps(self, boxes: List[Box]) -> List[Box]:
        boxes = sorted(boxes, key=lambda b: -b.score)

        # Per-class caps
        per_class_cap = {
            s.name: s.max_boxes for s in self.class_specs if s.max_boxes is not None
        }
        if per_class_cap:
            counts: Dict[str, int] = {}
            capped: List[Box] = []
            for b in boxes:
                cap = per_class_cap.get(b.class_name)
                if cap is not None and counts.get(b.class_name, 0) >= cap:
                    continue
                counts[b.class_name] = counts.get(b.class_name, 0) + 1
                capped.append(b)
            boxes = capped

        if self.max_boxes_total is not None:
            boxes = boxes[: self.max_boxes_total]
        return boxes

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cls_str = ", ".join(
            f"{s.name}({s.box_w}×{s.box_h})" for s in self.class_specs
        )
        return (
            f"MultiClassBBoxProposer(classes=[{cls_str}], "
            f"norm={self.normalization}, nms_iou={self.nms_iou_thresh}, "
            f"min_cov={self.min_coverage}, stride_frac={self.tile_stride_frac})"
        )
