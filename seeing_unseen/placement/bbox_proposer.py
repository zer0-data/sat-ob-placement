"""
BBoxProposer
============
Converts the affordance heatmap output of RemoteCLIPUNet into a ranked list
of placement bounding boxes.

Algorithm
---------
1. Gaussian-smooth the heatmap to suppress noise.
2. Find the top-K local maxima as candidate centres (with minimum inter-peak
   distance to avoid clustering on the same blob).
3. For each centre, generate:
     - one on-centre box
     - ``n_jitter`` boxes with random offsets up to ``jitter_frac`` × box size
4. Score every candidate box by the mean (or max/sum) heatmap value inside it.
5. Run NMS (IoU threshold configurable) on the full candidate set.
6. Return the top-``n_boxes`` survivors, sorted by score descending.

Usage
-----
    proposer = BBoxProposer(box_w=64, box_h=32, n_boxes=5)
    boxes = proposer.propose(heatmap)   # heatmap: (1, H, W) tensor in [0,1]
    # boxes -> [(x1, y1, x2, y2, score), ...]
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    from scipy.ndimage import gaussian_filter, maximum_filter
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    from torchvision.ops import nms as tv_nms
    _TV_NMS = True
except ImportError:
    _TV_NMS = False


# ---------------------------------------------------------------------------
# BBoxProposer
# ---------------------------------------------------------------------------

class BBoxProposer:
    """
    Parameters
    ----------
    box_w, box_h        : bounding box dimensions in pixels (user-supplied).
    n_boxes             : number of boxes to return after NMS.
    n_peaks             : how many local-maxima to seed candidates from.
    n_jitter            : off-centre jittered candidates generated per peak.
    jitter_frac         : jitter magnitude as a fraction of box dimensions.
                          E.g. 0.5 → offsets up to ±(box_w/2, box_h/2).
    smooth_sigma        : Gaussian sigma applied to heatmap before peak detection.
                          Higher = smoother, broader peaks.
    nms_iou_thresh      : IoU threshold for NMS suppression.
                          Lower = stricter (fewer boxes survive).
    min_peak_dist       : minimum pixel distance between two accepted peaks.
                          Prevents multiple centres inside the same blob.
    score_fn            : "mean" | "max" | "sum" — how to score a candidate box
                          from the heatmap values inside it.
    seed                : optional RNG seed for reproducible jitter.
    """

    def __init__(
        self,
        box_w: int,
        box_h: int,
        n_boxes: int = 5,
        n_peaks: int = 20,
        n_jitter: int = 8,
        jitter_frac: float = 0.5,
        smooth_sigma: float = 3.0,
        nms_iou_thresh: float = 0.3,
        min_peak_dist: int = 10,
        score_fn: str = "mean",
        seed: Optional[int] = None,
    ) -> None:
        if box_w <= 0 or box_h <= 0:
            raise ValueError(f"box_w and box_h must be positive, got {box_w}×{box_h}")
        if n_boxes <= 0:
            raise ValueError(f"n_boxes must be positive, got {n_boxes}")

        self.box_w = int(box_w)
        self.box_h = int(box_h)
        self.n_boxes = int(n_boxes)
        self.n_peaks = int(n_peaks)
        self.n_jitter = int(n_jitter)
        self.jitter_frac = float(jitter_frac)
        self.smooth_sigma = float(smooth_sigma)
        self.nms_iou_thresh = float(nms_iou_thresh)
        self.min_peak_dist = int(min_peak_dist)
        self.score_fn = score_fn
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose(
        self,
        heatmap: torch.Tensor,
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Parameters
        ----------
        heatmap : (1, H, W) or (H, W) float32 tensor, values in [0, 1].
                  Typically ``torch.sigmoid(model_output["affordance"])``
                  optionally multiplied by the validity mask.

        Returns
        -------
        boxes : list of (x1, y1, x2, y2, score) tuples, sorted by score
                descending.  Coordinates are in pixel space, clipped to image
                bounds.  len(boxes) == min(n_boxes, survivors after NMS).
        """
        # Normalise to (H, W) numpy float32
        hm: np.ndarray = heatmap.squeeze().detach().cpu().float().numpy()
        H, W = hm.shape

        # Guard: image must be large enough to fit at least one box
        if H < self.box_h or W < self.box_w:
            raise ValueError(
                f"Image ({H}×{W}) is smaller than requested box "
                f"({self.box_h}×{self.box_w})."
            )

        # 1. Smooth heatmap
        hm_smooth = self._smooth(hm)

        # 2. Find top-K local maxima
        peaks = self._find_peaks(hm_smooth, H, W)  # list of (row, col)

        if not peaks:
            return []

        # 3. Generate on-centre + jittered candidate boxes
        candidates = self._generate_candidates(peaks, H, W)

        if not candidates:
            return []

        # 4. Score each candidate by heatmap content
        scores = self._score_boxes(candidates, hm)

        # 5. NMS
        boxes_t = torch.tensor(candidates, dtype=torch.float32)
        scores_t = torch.tensor(scores, dtype=torch.float32)
        keep = self._nms(boxes_t, scores_t)

        # 6. Return top-n_boxes
        result = []
        for idx in keep[: self.n_boxes]:
            x1, y1, x2, y2 = candidates[idx]
            result.append((x1, y1, x2, y2, float(scores[idx])))

        return result

    # ------------------------------------------------------------------
    # Step 1 — Smoothing
    # ------------------------------------------------------------------

    def _smooth(self, hm: np.ndarray) -> np.ndarray:
        if not _SCIPY:
            # Fallback: manual separable box blur via convolution
            return hm
        if self.smooth_sigma <= 0:
            return hm
        return gaussian_filter(hm, sigma=self.smooth_sigma)

    # ------------------------------------------------------------------
    # Step 2 — Peak detection
    # ------------------------------------------------------------------

    def _find_peaks(
        self, hm_smooth: np.ndarray, H: int, W: int
    ) -> List[Tuple[int, int]]:
        """Return up to n_peaks (row, col) local maxima, sorted by value desc."""

        if _SCIPY:
            footprint = max(3, self.min_peak_dist * 2 + 1)
            local_max = maximum_filter(hm_smooth, size=footprint)
            peak_mask = (hm_smooth == local_max) & (hm_smooth > 0)
            rows, cols = np.where(peak_mask)
        else:
            # Fallback: sliding max via stride tricks (no scipy)
            rows, cols = self._find_peaks_numpy(hm_smooth)

        if len(rows) == 0:
            return []

        # Sort by descending value
        vals = hm_smooth[rows, cols]
        order = np.argsort(-vals)
        rows, cols = rows[order], cols[order]

        # Greedy suppression: enforce min_peak_dist between selected peaks
        selected: List[Tuple[int, int]] = []
        for r, c in zip(rows.tolist(), cols.tolist()):
            if len(selected) >= self.n_peaks:
                break
            too_close = any(
                abs(r - sr) < self.min_peak_dist and abs(c - sc) < self.min_peak_dist
                for sr, sc in selected
            )
            if not too_close:
                selected.append((int(r), int(c)))

        return selected

    def _find_peaks_numpy(
        self, hm: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pure-numpy fallback peak finder (slower but no scipy dependency)."""
        H, W = hm.shape
        d = self.min_peak_dist
        rows, cols = [], []
        for r in range(0, H, max(1, d)):
            for c in range(0, W, max(1, d)):
                r0, r1 = max(0, r - d), min(H, r + d + 1)
                c0, c1 = max(0, c - d), min(W, c + d + 1)
                patch = hm[r0:r1, c0:c1]
                lr, lc = np.unravel_index(patch.argmax(), patch.shape)
                rows.append(r0 + lr)
                cols.append(c0 + lc)
        return np.array(rows), np.array(cols)

    # ------------------------------------------------------------------
    # Step 3 — Candidate generation
    # ------------------------------------------------------------------

    def _generate_candidates(
        self,
        peaks: List[Tuple[int, int]],
        H: int,
        W: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        For each peak:
          - 1 on-centre box
          - n_jitter boxes with random pixel offsets bounded by jitter_frac
        """
        bw, bh = self.box_w, self.box_h
        half_w = bw // 2
        half_h = bh // 2

        max_jx = max(1, int(bw * self.jitter_frac))
        max_jy = max(1, int(bh * self.jitter_frac))

        # Upper-left coordinate limits so boxes stay inside the image
        max_x1 = W - bw
        max_y1 = H - bh

        candidates: List[Tuple[int, int, int, int]] = []

        for row, col in peaks:
            # All offsets for this peak: (0,0) first, then random jitter
            jx = self._rng.integers(-max_jx, max_jx + 1, size=self.n_jitter)
            jy = self._rng.integers(-max_jy, max_jy + 1, size=self.n_jitter)
            offsets = [(0, 0)] + list(zip(jx.tolist(), jy.tolist()))

            for dx, dy in offsets:
                cx = col + dx
                cy = row + dy
                x1 = int(np.clip(cx - half_w, 0, max_x1))
                y1 = int(np.clip(cy - half_h, 0, max_y1))
                x2 = x1 + bw
                y2 = y1 + bh
                candidates.append((x1, y1, x2, y2))

        return candidates

    # ------------------------------------------------------------------
    # Step 4 — Scoring
    # ------------------------------------------------------------------

    def _score_boxes(
        self,
        candidates: List[Tuple[int, int, int, int]],
        hm: np.ndarray,
    ) -> List[float]:
        scores = []
        for x1, y1, x2, y2 in candidates:
            region = hm[y1:y2, x1:x2]
            if region.size == 0:
                scores.append(0.0)
            elif self.score_fn == "max":
                scores.append(float(region.max()))
            elif self.score_fn == "sum":
                scores.append(float(region.sum()))
            else:  # default: mean
                scores.append(float(region.mean()))
        return scores

    # ------------------------------------------------------------------
    # Step 5 — NMS
    # ------------------------------------------------------------------

    def _nms(
        self,
        boxes: torch.Tensor,   # (N, 4) — x1,y1,x2,y2
        scores: torch.Tensor,  # (N,)
    ) -> List[int]:
        if _TV_NMS:
            keep = tv_nms(boxes, scores, self.nms_iou_thresh)
            return keep.tolist()
        return self._nms_python(
            boxes.numpy().astype(np.float32),
            scores.numpy().astype(np.float32),
        )

    def _nms_python(
        self, boxes: np.ndarray, scores: np.ndarray
    ) -> List[int]:
        """Pure-numpy greedy NMS fallback."""
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
    # Convenience helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BBoxProposer(box={self.box_w}×{self.box_h}, "
            f"n_boxes={self.n_boxes}, n_peaks={self.n_peaks}, "
            f"n_jitter={self.n_jitter}, jitter_frac={self.jitter_frac}, "
            f"nms_iou={self.nms_iou_thresh})"
        )
