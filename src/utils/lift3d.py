"""3D stages of the pipeline: lift masks to 3D boxes, plus the ground truth the evaluation needs.

  compute_boxes     per mask (row-aligned with the masks stage): world-frame 3D box
  compute_events    GT only: when each object is hidden (leaves view) and revisited (comes back)
  compute_gt_boxes  GT only: one fused 3D box per object, to score the predicted boxes against

Replica objects never move, so "hidden" means "not visible", either out of the field of view or occluded.
"""

import numpy as np
import pycocotools.mask as mask_utils
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion

from utils import replica
# from utils.events import extract_events
# from utils.geometry import backproject, mask_to_box, robust_filter
from utils.schema import Boxes, Events, ObjectBoxes, Run


def geometry_source(run: Run):
    """Returns get(frame) -> (depth (H, W) m, K (3, 3), c2w (4, 4)) for the run's pose_source.

    pose_source "vggt" plugs in here: it must supply depth, intrinsics and poses with this same signature.
    """
    scene, seq = replica.parse_source(run.source)
    if run.pose_source != "gt":
        raise NotImplementedError(f"pose_source {run.pose_source!r}: depth, K and poses come from the VGGT stage")
    poses = replica.load_poses(scene, seq)
    return lambda i: (replica.load_depth(scene, seq, i), replica.K, poses[i])


def compute_boxes(run: Run, masks) -> Boxes:
    """One 3D box per mask. Rows with too little valid depth get NaN and n_points 0."""
    get = geometry_source(run)
    n = len(masks.frame)
    box_min = np.full((n, 3), np.nan, np.float32)
    box_max = np.full((n, 3), np.nan, np.float32)
    n_points = np.zeros(n, np.int32)
    for frame in np.unique(masks.frame): # process all masks per frame together (shared frame info)
        rows = np.nonzero(masks.frame == frame)[0]
        depth, K, c2w = get(int(frame))
        stack = mask_utils.decode([masks.rle[r] for r in rows]).astype(bool)  # (H, W, k)
        if stack.shape[:2] != depth.shape:
            raise ValueError(f"frame {frame}: masks are {stack.shape[:2]} but depth is {depth.shape}")
        for k, r in enumerate(rows):
            box = mask_to_box(depth, K, c2w, stack[:, :, k])
            if box is not None:
                box_min[r], box_max[r], n_points[r] = box.min, box.max, box.n_points
    return Boxes(box_min, box_max, n_points)

def rle_area(rles: list[dict], chunk=255) -> np.ndarray:
    """Pixel count of each RLE mask. pycocotools.mask.area fails on NumPy 2 for more than 255 masks at once."""
    if not rles:
        return np.zeros(0, np.int64)
    return np.concatenate([mask_utils.area(rles[i : i + chunk]) for i in range(0, len(rles), chunk)]).astype(np.int64)

def visibility_pixels(run: Run):
    """GT visibility: (frames (F,), pixels (F, n_ids)) where pixels[f, id] is the object's pixel count."""
    masks, _ = replica.gt_masks(run)
    frames = np.array(replica.frame_numbers(run.stride))
    pixels = np.zeros((len(frames), int(masks.sam_id.max()) + 1 if len(masks.sam_id) else 1), np.int32)
    if len(masks.frame):
        pixels[masks.frame // run.stride, masks.sam_id] = rle_area(masks.rle)
    return frames, pixels


def compute_events(run: Run) -> Events:
    frames, pixels = visibility_pixels(run)
    events, _ = extract_events(pixels, frames, MIN_PIXELS, MIN_RUN, MIN_GAP)
    return Events(
        object_id=np.array([e["object_id"] for e in events], np.int32),
        frame=np.array([e["frame"] for e in events], np.int32),
        kind=np.array([KIND[e["type"]] for e in events], np.int8),
        gap=np.array([e.get("gap_frames", 0) for e in events], np.int32),
    )


def compute_gt_boxes(run: Run, sample_per_frame=2000, seed=0) -> ObjectBoxes:
    """Fuse each object's GT points over all frames and take a robust box."""
    scene, seq = replica.parse_source(run.source)
    get = geometry_source(run)
    rng = np.random.default_rng(seed)
    chunks: dict[int, list] = {}
    for i in replica.frame_numbers(run.stride):
        depth, K, c2w = get(i)
        ids = replica.load_instance_ids(scene, seq, i)
        for oid in np.unique(ids):
            if oid == 0:
                continue
            pts, _ = backproject(depth, K, c2w, ids == oid)
            if len(pts) < 200:
                continue
            if len(pts) > sample_per_frame:
                pts = pts[rng.choice(len(pts), sample_per_frame, replace=False)]
            chunks.setdefault(int(oid), []).append(pts)

    rows = []
    for oid in sorted(chunks):
        pts = robust_filter(np.concatenate(chunks[oid]), mad_scale=4.0)
        if len(pts) >= 50:
            rows.append((oid, pts.min(0), pts.max(0), len(pts)))
    return ObjectBoxes(
        object_id=np.array([r[0] for r in rows], np.int32),
        box_min=np.array([r[1] for r in rows], np.float32).reshape(-1, 3),
        box_max=np.array([r[2] for r in rows], np.float32).reshape(-1, 3),
        n_points=np.array([r[3] for r in rows], np.int32),
    )

# GEOMETRY HELPERS

def backproject(depth, K, c2w, mask=None, max_depth=10.0):
    """Lift pixels to world-space points. Returns (N, 3) points and the (N, 2) pixel coords (v, u)."""
    valid = (depth > 0) & (depth < max_depth)
    if mask is not None:
        valid &= mask
    v, u = np.nonzero(valid)
    z = depth[v, u]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    return (cam @ c2w.T)[:, :3], np.stack([v, u], axis=1)


def robust_filter(points, mad_scale=3.0):
    """Drop outliers: keep points within mad_scale * MAD of the median, per axis.
    Handles depth bleeding at mask edges ("flying pixels") and wrong mask pixels."""
    if len(points) < 10:
        return points
    med = np.median(points, axis=0)
    mad = np.median(np.abs(points - med), axis=0) * 1.4826 + 1e-3  # +1 mm floor
    keep = np.all(np.abs(points - med) <= mad_scale * mad, axis=1)
    return points[keep]


@dataclass
class Box3D:
    min: np.ndarray  # (3,)
    max: np.ndarray  # (3,)
    n_points: int

    @property
    def center(self):
        return (self.min + self.max) / 2

    @property
    def size(self):
        return self.max - self.min

    @property
    def volume(self):
        return float(np.prod(np.maximum(self.size, 0)))


def mask_to_box(depth, K, c2w, mask, erode_px=2, min_points=50, mad_scale=3.0, max_depth=10.0):
    """Axis-aligned world-frame box of a mask, or None if too few valid depth pixels."""
    if erode_px > 0:
        eroded = binary_erosion(mask, iterations=erode_px)
        if eroded.sum() >= min_points:  # don't erode small objects away
            mask = eroded
    pts, _ = backproject(depth, K, c2w, mask, max_depth)
    pts = robust_filter(pts, mad_scale)
    if len(pts) < min_points:
        return None
    return Box3D(pts.min(0), pts.max(0), len(pts))


def box_iou_3d(a: Box3D, b: Box3D) -> float:
    lo, hi = np.maximum(a.min, b.min), np.minimum(a.max, b.max)
    inter = float(np.prod(np.maximum(hi - lo, 0)))
    union = a.volume + b.volume - inter
    return inter / union if union > 0 else 0.0


# EVENTS: keep track of objects when they leave and re-appear in the scene

# Thresholds in raw frames, so they do not change with the run's stride
MIN_PIXELS = 500  # an object counts as visible in a frame above this many pixels
MIN_RUN = 50  # shorter visible runs are ignored as flicker
MIN_GAP = 60  # hidden at least this long before a return counts as a revisit

KIND = {"hidden": 0, "revisited": 1}

def extract_events(pixels, frame_indices, min_pixels=500, min_run=10, min_gap=60, ignore_ids=(0,)):
    """Per object: visible runs and hide/revisit events.

    min_pixels: pixel count for the object to count as visible in a frame
    min_run:    a visible run shorter than this is ignored (flicker)
    min_gap:    hidden at least this long before a return counts as a revisit
    min_run and min_gap are in raw frames, so they don't depend on the stride.
    """
    frame_indices = np.asarray(frame_indices)
    step = int(frame_indices[1] - frame_indices[0]) if len(frame_indices) > 1 else 1
    min_run = int(np.ceil(min_run / step))   # in exported frames from here on
    min_gap = int(np.ceil(min_gap / step))
    events, summary = [], {}
    for o in range(pixels.shape[1]):
        if o in ignore_ids:
            continue
        vis = pixels[:, o] >= min_pixels
        runs = _runs(vis)
        runs = [(s, e) for s, e in runs if e - s + 1 >= min_run]
        # merge runs separated by a short gap: not a real hide
        merged = []
        for s, e in runs:
            if merged and s - merged[-1][1] - 1 < min_gap:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        if not merged:
            continue
        summary[o] = {"runs": [(int(frame_indices[s]), int(frame_indices[e])) for s, e in merged]}
        for (_, e1), (s2, _) in zip(merged[:-1], merged[1:]):
            events.append({"object_id": int(o), "type": "hidden", "frame": int(frame_indices[e1])})
            events.append({"object_id": int(o), "type": "revisited", "frame": int(frame_indices[s2]),
                           "gap_frames": int(frame_indices[s2] - frame_indices[e1])})
    events.sort(key=lambda ev: (ev["frame"], ev["object_id"]))
    return events, summary


def _runs(mask):
    """Inclusive (start, end) index pairs of True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1))