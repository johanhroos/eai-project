"""Each object's box from every view of it: around the points its views agree on, so that no one view's depth sets it.

Each of a node's reliable masks (lift3d.reliable) is lifted again here, from its frame's depth at the depth's own
resolution: every pixel the mask covers more than COVER of, without the outliers of lift3d.robust_filter. Each view
then weighs by its pixels on the object, so close views count more than far ones, and the box can reach the object's
ends, which the boxes stage's sample of lift3d.POINTS_PER_MASK points per mask, from the depth upsampled to the image,
often misses. A mask too small for MIN_POINTS points at that resolution gives that sample instead.

Pooled, those points make up the object, plus wherever a view's depth was wrong: a far view's depth smears along its
camera ray, and one streak makes the box metres long while the point cloud, which is judged by its bulk, still looks
right. The node's other views catch such points. Projected into another frame the node is seen in, a point of the
object lands inside the node's mask there, or behind something in front of it. A point is kept when it lands inside
the mask in at least SUPPORT of the views where it is in the image and not hidden; a streak from one view lands outside
the mask in the views from other directions. The box is around the kept points without the outer TRIM percent on each
axis.

On the 8 Replica scenes, over the nodes of objects other than walls, floors and ceilings, these boxes overlap the GT
objects' boxes more than graph.nodes.node_box's around all the pooled points: IoU 0.38 to 0.45 with VGGT-Omega's depth
(both boxes grown by 5 cm on each side), 0.41 to 0.51 with Replica's, and 0.55 to 0.61 with GT masks and VGGT-Omega's
depth, so most of what is left is the grouping and the depth. Voting with the boxes stage's samples gave 0.43, 0.48
and 0.58. Lifting like views/rerun_map.py's point cloud, without depth edges and thinned to one point per 2 cm but
without robust_filter, gave 0.41: the map's cloud has the same streaks, it only shows them as a few specks among the
surfaces, and thinning gives those specks as much weight as the surfaces.

The graph stage keeps these boxes (Graph.box_min and box_max), and graph.nodes.build_nodes uses them. The object
memory can't: it only knows the frames so far, and boxes around the pooled points (graph.nodes.node_box) are what it
matches with.
"""

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from data.stages import Boxes, Masks, Run, load_stage
from geometry import lift3d
from graph.nodes import node_box

COVER = 0.5  # share of a pixel at the depth's resolution a mask must cover for the pixel to be lifted
MAD = 3.0  # robust_filter's limit for each view's points, in median absolute deviations
MIN_POINTS = 50  # fewer lifted points than this, and a mask gives the boxes stage's sample of its points
SUPPORT = 0.5  # share of the views a point is in where it must land inside the node's mask
HIDDEN = (0.05, 0.08)  # metres and share of the depth: a point further behind the depth there is hidden in a view
TRIM = 0.5  # percent of the kept points left out on each side of each axis
MAX_POINTS = 8192  # of a node's pooled points that vote, spread over them
PARAMS = {
    "cover": COVER,
    "mad": MAD,
    "min_points": MIN_POINTS,
    "support": SUPPORT,
    "hidden": list(HIDDEN),
    "trim": TRIM,
    "max_points": MAX_POINTS,
}


def view_source(run: Run):
    """Returns get(frame) -> (depth (h, w) m, K (3, 3) for that size, c2w (4, 4)) at the depth's own resolution.

    As lift3d.geometry_source, but at VGGT-Omega's model resolution, or every second pixel of Replica's GT depth.
    """
    if run.pose_source != "vggt":
        get = lift3d.geometry_source(run)

        def every_second(i):
            depth, K, c2w = get(i)
            return depth[::2, ::2], np.diag([0.5, 0.5, 1.0]) @ K, c2w

        return every_second
    from data import footage
    from data.stages import Geometry
    from models.vggt import world_from_camera

    geometry = load_stage(run.stage_dir("vggt"), Geometry)
    row = {int(f): k for k, f in enumerate(geometry.frame)}
    height, width = footage.image_size(run)
    h, w = geometry.depth.shape[1:]

    def get(i):
        k = row[i]
        K = geometry.intrinsics[k].astype(np.float64).copy()
        K[0] *= w / width
        K[1] *= h / height
        return geometry.depth[k].astype(np.float32) * geometry.scale, K, world_from_camera(geometry, k)

    return get


def shrink(mask: np.ndarray, size: tuple[int, int], cover: float = 0.25) -> np.ndarray:
    """A (H, W) bool mask at (h, w): the pixels it covers more than cover of. A quarter keeps thin parts in it."""
    small = Image.fromarray(mask.astype(np.uint8) * 255).resize((size[1], size[0]), Image.Resampling.BILINEAR)
    return np.asarray(small) > round(cover * 255)


def object_boxes(run: Run, masks: Masks, boxes: Boxes, node: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(M, 3) box_min and box_max of every node ID 0 ... M - 1 in node, the node ID of every mask (-1 for none).

    A node without a reliable mask gets graph.nodes.node_box's box, and IDs no mask has get NaN.
    """
    n_ids = int(node.max()) + 1 if len(node) and node.max() >= 0 else 0
    lo, hi = np.full((n_ids, 3), np.nan, np.float32), np.full((n_ids, 3), np.nan, np.float32)
    kept = object_points(run, masks, boxes, node)
    for n in np.unique(node[node >= 0]).tolist():
        if n in kept:
            lo[n], hi[n] = np.percentile(kept[n], (TRIM, 100 - TRIM), axis=0)
        else:
            lo[n], hi[n] = node_box(boxes, np.flatnonzero(node == n))
    return lo, hi


def object_points(
    run: Run, masks: Masks, boxes: Boxes, node: np.ndarray, max_points: int | None = MAX_POINTS, voxel: float = 0.0
) -> dict[int, np.ndarray]:
    """The points of each node with a reliable mask that its views agree on, (k, 3) float32 by node ID.

    node is the node ID of every mask (-1 for none). Each node's pooled points are thinned to one per voxel-sided cube
    (0: not thinned), then at most max_points of them (None: all), spread over them, vote. When fewer than 10 are kept,
    all of them are returned. object_boxes draws the boxes around these; evaluate_openlex3d.py takes them as the
    objects' point clouds.
    """
    get = view_source(run)
    views = {}  # frame -> its depth, K and c2w
    points = {}  # row of a reliable mask -> its points (k, 3)
    lifted = np.flatnonzero((node >= 0) & lift3d.reliable(boxes))
    for frame in np.unique(masks.frame[lifted]).tolist():
        rows = lifted[masks.frame[lifted] == frame]
        depth, K, c2w = views[frame] = get(frame)
        stack = mask_utils.decode([masks.rle[r] for r in rows]).astype(bool)  # (H, W, k)
        for k, r in enumerate(rows.tolist()):
            mask = shrink(stack[:, :, k], depth.shape, COVER)
            p = lift3d.robust_filter(lift3d.backproject(depth, K, c2w, mask)[0], MAD)
            if len(p) < MIN_POINTS and boxes.points is not None:
                p = boxes.points[r].astype(np.float32)
                p = p[np.isfinite(p).all(axis=1)]
            points[r] = p.astype(np.float32)

    votes = {}  # node ID -> its points (k, 3), the frame each is from (k,), and its views' votes for and against
    for n in np.unique(node[node >= 0]).tolist():
        rows = [r for r in np.flatnonzero(node == n).tolist() if len(points.get(r, ()))]
        if not rows:
            continue
        pooled = np.concatenate([points[r] for r in rows])
        source = np.concatenate([np.full(len(points[r]), masks.frame[r]) for r in rows])
        if voxel > 0:  # the first point in each cube, so that it keeps the frame it is from
            cube = np.floor(pooled / voxel).astype(np.int64)
            _, first = np.unique(cube, axis=0, return_index=True)
            pooled, source = pooled[np.sort(first)], source[np.sort(first)]
        n_vote = len(pooled) if max_points is None else min(len(pooled), max_points)
        spread = np.unique(np.linspace(0, len(pooled) - 1, n_vote).astype(int))
        votes[n] = (pooled[spread], source[spread], np.zeros(len(spread)), np.zeros(len(spread)))

    for frame in np.unique(masks.frame).tolist():
        rows = np.flatnonzero((masks.frame == frame) & np.isin(node, list(votes)))
        if not len(rows):
            continue
        depth, K, c2w = views[frame] if frame in views else get(frame)
        h, w = depth.shape
        w2c = np.linalg.inv(c2w)
        stack = mask_utils.decode([masks.rle[r] for r in rows]).astype(bool)  # (H, W, k)
        for n in np.unique(node[rows]).tolist():
            mask = shrink(stack[:, :, node[rows] == n].any(axis=2), (h, w))
            pooled, source, agree, disagree = votes[n]
            cam = pooled @ w2c[:3, :3].T + w2c[:3, 3]
            z = np.maximum(cam[:, 2], 1e-6)
            u = np.round(K[0, 0] * cam[:, 0] / z + K[0, 2]).astype(int)
            v = np.round(K[1, 1] * cam[:, 1] / z + K[1, 2]).astype(int)
            seen = (cam[:, 2] > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (source != frame)
            u, v = np.where(seen, u, 0), np.where(seen, v, 0)
            inside = seen & mask[v, u]
            hidden = cam[:, 2] > depth[v, u] + np.maximum(HIDDEN[0], HIDDEN[1] * depth[v, u])
            agree += inside
            disagree += seen & ~inside & ~hidden

    kept_points = {}
    for n, (pooled, _, agree, disagree) in votes.items():
        kept = agree >= SUPPORT * (agree + disagree)  # a point no other view sees is kept
        kept_points[n] = pooled[kept] if kept.sum() >= 10 else pooled
    return kept_points
