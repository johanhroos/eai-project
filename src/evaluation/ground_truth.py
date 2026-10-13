"""Ground truth for the evaluation, always from Replica's masks, depth and poses, whatever the run's sources are.

  compute_events    when each object is hidden (leaves the view) and revisited (comes back)
  compute_gt_boxes  one fused 3D box per object, to score the predicted boxes against

Replica objects never move, so "hidden" means "not visible", either out of the field of view or occluded.
"""

from dataclasses import replace
from itertools import pairwise

import numpy as np

from data import replica
from data.stages import Events, ObjectBoxes, Run, rle_area
from geometry.depth import backproject, geometry_source, robust_filter

# Thresholds in raw frames, so they do not change with the run's stride
MIN_PIXELS = 500  # an object counts as visible in a frame above this many pixels
MIN_RUN = 50  # shorter visible runs are ignored as flicker
MIN_GAP = 60  # hidden at least this long before a return counts as a revisit
KIND = {"hidden": 0, "revisited": 1}


def compute_events(run: Run) -> Events:
    frames, pixels = visibility_pixels(run)
    events = extract_events(pixels, frames, MIN_PIXELS, MIN_RUN, MIN_GAP)
    return Events(
        object_id=np.array([e["object_id"] for e in events], np.int32),
        frame=np.array([e["frame"] for e in events], np.int32),
        kind=np.array([KIND[e["type"]] for e in events], np.int8),
        gap=np.array([e.get("gap_frames", 0) for e in events], np.int32),
    )


def visibility_pixels(run: Run):
    """GT visibility: (frames (F,), pixels (F, n_ids)) where pixels[f, id] is the object's pixel count."""
    masks, _ = replica.gt_masks(run)
    frames = np.array(replica.frame_numbers(run.stride))
    pixels = np.zeros((len(frames), int(masks.sam_id.max()) + 1 if len(masks.sam_id) else 1), np.int32)
    if len(masks.frame):
        pixels[masks.frame // run.stride, masks.sam_id] = rle_area(masks.rle)
    return frames, pixels


def extract_events(pixels, frame_indices, min_pixels=500, min_run=10, min_gap=60, ignore_ids=(0,)):
    """Hide and revisit events of every object, in frame order.

    min_pixels: pixel count for the object to count as visible in a frame
    min_run:    a visible run shorter than this is ignored (flicker)
    min_gap:    hidden at least this long before a return counts as a revisit
    min_run and min_gap are in raw frames, so they don't depend on the stride.
    """
    frame_indices = np.asarray(frame_indices)
    step = int(frame_indices[1] - frame_indices[0]) if len(frame_indices) > 1 else 1
    min_run = int(np.ceil(min_run / step))  # in exported frames from here on
    min_gap = int(np.ceil(min_gap / step))
    events = []
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
        for (_, e1), (s2, _) in pairwise(merged):
            events.append({"object_id": int(o), "type": "hidden", "frame": int(frame_indices[e1])})
            gap = int(frame_indices[s2] - frame_indices[e1])
            events.append(
                {"object_id": int(o), "type": "revisited", "frame": int(frame_indices[s2]), "gap_frames": gap}
            )
    events.sort(key=lambda ev: (ev["frame"], ev["object_id"]))
    return events


def _runs(mask):
    """Inclusive (start, end) index pairs of True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1, strict=True))


def compute_gt_boxes(run: Run, sample_per_frame=2000, seed=0) -> ObjectBoxes:
    """Fuse each object's GT points over all frames and take a robust box, always from GT depth and poses."""
    scene, seq = replica.parse_source(run.source)
    get = geometry_source(replace(run, pose_source="gt"))  # the stage is shared by every pose_source
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
