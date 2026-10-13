"""Depth and cameras of a run's frames, and lifting pixels to world points.

The depth and poses are Replica's (pose_source "gt") or VGGT-Omega's (the vggt stage, models/vggt.py), in metres in a
world frame with z up either way.
"""

import numpy as np

from data import footage, replica
from data.stages import Geometry, Run, load_stage


def geometry_source(run: Run, full_size: bool = True):
    """Returns get(frame) -> (depth (h, w) m, K (3, 3) for that size, c2w (4, 4)) for the run's pose_source.

    "gt" reads Replica's depth and poses. "vggt" reads the vggt stage: the model's depth and its cameras, both in metres
    with z up through the stage's Sim(3). It runs on any source. full_size gives the depth at the image's size (VGGT-
    Omega's upsampled); otherwise at the depth's own resolution: VGGT-Omega's, or every second pixel of Replica's.
    """
    if run.pose_source == "gt":
        scene, seq = replica.parse_source(run.source)
        poses = replica.load_poses(scene, seq)
        if full_size:
            return lambda i: (replica.load_depth(scene, seq, i), replica.K, poses[i])
        half = np.diag([0.5, 0.5, 1.0]) @ replica.K
        return lambda i: (replica.load_depth(scene, seq, i)[::2, ::2], half, poses[i])
    if run.pose_source != "vggt":
        raise ValueError(f"unknown pose_source {run.pose_source!r}")
    from models.vggt import upsample, world_from_camera  # imports VGGT-Omega only for runs that use it

    geometry = load_stage(run.stage_dir("vggt"), Geometry)
    row = {int(f): k for k, f in enumerate(geometry.frame)}
    height, width = footage.image_size(run)
    h, w = geometry.depth.shape[1:]

    def get(i):
        k = row[i]
        K = geometry.intrinsics[k].astype(np.float64)
        if full_size:
            return upsample(geometry.depth[k], (height, width)) * geometry.scale, K, world_from_camera(geometry, k)
        K[0] *= w / width
        K[1] *= h / height
        return geometry.depth[k].astype(np.float32) * geometry.scale, K, world_from_camera(geometry, k)

    return get


def confidence_source(run: Run):
    """Returns get(frame) -> (H, W) confidence of the depth at the image size, or None when the depth has none.

    VGGT-Omega's confidence is 1 + exp(x): at least 1, and higher is better. Replica's GT depth has none.
    """
    if run.pose_source != "vggt":
        return None
    from models.vggt import upsample

    geometry = load_stage(run.stage_dir("vggt"), Geometry)
    row = {int(f): k for k, f in enumerate(geometry.frame)}
    size = footage.image_size(run)
    return lambda i: upsample(geometry.conf[row[i]], size)


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

    Handles depth bleeding at mask edges ("flying pixels") and wrong mask pixels.
    """
    if len(points) < 10:
        return points
    med = np.median(points, axis=0)
    mad = np.median(np.abs(points - med), axis=0) * 1.4826 + 1e-3  # +1 mm floor
    keep = np.all(np.abs(points - med) <= mad_scale * mad, axis=1)
    return points[keep]
