"""The boxes stage: a world-frame 3D box of every mask (row-aligned with the masks stage), from its frame's depth.

Each box keeps a sample of POINTS_PER_MASK of the mask's points, so that an object's box can later be fit to all its
views' points, and the median confidence of the depth under the mask: reliable says which masks' depth to trust.
"""

from dataclasses import dataclass

import numpy as np
import pycocotools.mask as mask_utils
from scipy.ndimage import binary_erosion

from data.stages import Boxes, Run
from geometry.depth import backproject, confidence_source, geometry_source, robust_filter

POINTS_PER_MASK = 256  # kept by compute_boxes, so that an object's box can be fit to all its views' points
MIN_CONFIDENCE = 2.5  # median VGGT-Omega depth confidence under which a mask's depth is not trusted (reliable)


@dataclass
class Box3D:
    min: np.ndarray  # (3,)
    max: np.ndarray  # (3,)
    n_points: int
    points: np.ndarray | None = None  # (n_points, 3), the points the box is drawn around
    conf: float = float("nan")  # median confidence of the depth under the mask, NaN without one

    @property
    def center(self):
        return (self.min + self.max) / 2

    @property
    def size(self):
        return self.max - self.min

    @property
    def volume(self):
        return float(np.prod(np.maximum(self.size, 0)))


def mask_to_box(depth, K, c2w, mask, conf=None, erode_px=2, min_points=50, mad_scale=3.0, max_depth=10.0):
    """Axis-aligned world-frame box of a mask, or None if too few valid depth pixels.

    conf, the depth's (H, W) confidence if it has one, gives the box the median confidence of the mask's pixels.
    """
    if erode_px > 0:
        eroded = binary_erosion(mask, iterations=erode_px)
        if eroded.sum() >= min_points:  # don't erode small objects away
            mask = eroded
    pts, vu = backproject(depth, K, c2w, mask, max_depth)
    median_conf = float(np.median(conf[vu[:, 0], vu[:, 1]])) if conf is not None and len(vu) else float("nan")
    pts = robust_filter(pts, mad_scale)
    if len(pts) < min_points:
        return None
    return Box3D(pts.min(0), pts.max(0), len(pts), pts, median_conf)


def box_containment(a: Box3D, b: Box3D) -> float:
    """Share of the smaller box's volume inside the larger: 1 for a view of part of an object inside the object's box.

    IoU is near 0 there, since it divides by the larger box too. 0 when the boxes don't touch or one has no volume.
    """
    lo, hi = np.maximum(a.min, b.min), np.minimum(a.max, b.max)
    inter = float(np.prod(np.maximum(hi - lo, 0)))
    smaller = min(a.volume, b.volume)
    return inter / smaller if smaller > 0 else 0.0


def reliable(boxes: Boxes) -> np.ndarray:
    """(N,) bool: whether to trust each mask's depth.

    It needs a box, and a median confidence of at least MIN_CONFIDENCE where the depth has one. Far views fail: their
    depth smears along the camera ray.
    """
    ok = boxes.n_points > 0
    if boxes.conf is not None:
        ok &= ~(boxes.conf < MIN_CONFIDENCE)  # NaN (no confidence) passes
    return ok


def compute_boxes(run: Run, masks) -> Boxes:
    """One 3D box per mask, with a sample of POINTS_PER_MASK of its points and the median confidence of its depth.

    Rows with too little valid depth get NaN and n_points 0. conf is NaN for depth without a confidence (Replica's).
    """
    get = geometry_source(run)
    get_conf = confidence_source(run)
    rng = np.random.default_rng(0)
    n = len(masks.frame)
    box_min = np.full((n, 3), np.nan, np.float32)
    box_max = np.full((n, 3), np.nan, np.float32)
    n_points = np.zeros(n, np.int32)
    points = np.full((n, POINTS_PER_MASK, 3), np.nan, np.float16)
    conf = np.full(n, np.nan, np.float32)
    for frame in np.unique(masks.frame):  # all masks of a frame at once, which share its depth
        rows = np.nonzero(masks.frame == frame)[0]
        depth, K, c2w = get(int(frame))
        frame_conf = get_conf(int(frame)) if get_conf else None
        stack = mask_utils.decode([masks.rle[r] for r in rows]).astype(bool)  # (H, W, k)
        if stack.shape[:2] != depth.shape:
            raise ValueError(f"frame {frame}: masks are {stack.shape[:2]} but depth is {depth.shape}")
        for k, r in enumerate(rows):
            box = mask_to_box(depth, K, c2w, stack[:, :, k], frame_conf)
            if box is not None:
                box_min[r], box_max[r], n_points[r], conf[r] = box.min, box.max, box.n_points, box.conf
                keep = rng.choice(box.n_points, min(box.n_points, POINTS_PER_MASK), replace=False)
                points[r, : len(keep)] = box.points[keep]
    return Boxes(box_min, box_max, n_points, points, conf)
