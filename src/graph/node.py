"""Object nodes of the scene graph, put together from the graph stage's node ID per mask."""

from dataclasses import dataclass

import numpy as np

from utils.schema import Boxes, Features, Graph, Masks

# A node's box spans the 2nd to 98th percentile of its masks' sampled points on each axis, so a few stray points
# (a mask spilling onto the background) don't stretch it. Without point samples (older boxes stages), the 10th to 90th
# percentile of the masks' box corners, which comes close to the union of all views and is larger.
POINT_PERCENTILE = 2
FUSE_PERCENTILE = 10
# Before that, masks whose box centre is far from the median centre are left out: a SAM ID that jumped onto another
# object for a frame. Far is OUTLIER times the masks' median distance from it, and at least MIN_SPREAD metres.
OUTLIER = 3
MIN_SPREAD = 0.3


@dataclass
class Node:
    id: int
    label: str  # the prompt with the most SAM score summed over the node's masks
    box_min: np.ndarray  # (3,) float32, world frame, metres; NaN if none of the masks has a box
    box_max: np.ndarray  # (3,) float32
    dino: np.ndarray  # (D,) float32, mean of the masks' DINO embeddings, L2-normalized
    rows: np.ndarray  # (k,) int, the node's rows in the masks stage
    frames: np.ndarray  # (f,) int32, the frames it is seen in, sorted
    sam_ids: list[int]  # the SAM tracks merged into it


def fused_box(boxes: Boxes, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One box for the masks in rows, from their points (or per-mask boxes). NaN if none of them has a box."""
    rows = rows[boxes.n_points[rows] > 0]
    if not len(rows):
        return np.full(3, np.nan, np.float32), np.full(3, np.nan, np.float32)
    centre = (boxes.box_min[rows] + boxes.box_max[rows]) / 2
    distance = np.linalg.norm(centre - np.median(centre, axis=0), axis=1)
    rows = rows[distance <= max(OUTLIER * np.median(distance), MIN_SPREAD)]
    if boxes.points is not None:
        points = boxes.points[rows].reshape(-1, 3).astype(np.float32)
        points = points[np.isfinite(points[:, 0])]
        lo = np.percentile(points, POINT_PERCENTILE, axis=0)
        hi = np.percentile(points, 100 - POINT_PERCENTILE, axis=0)
    else:
        lo = np.percentile(boxes.box_min[rows], FUSE_PERCENTILE, axis=0)
        hi = np.percentile(boxes.box_max[rows], 100 - FUSE_PERCENTILE, axis=0)
    return lo.astype(np.float32), hi.astype(np.float32)


def mean_embedding(features: Features, rows: np.ndarray) -> np.ndarray:
    """L2-normalized mean of the DINO embeddings of the masks in rows."""
    e = features.dino[rows].astype(np.float32).sum(0)
    return e / max(float(np.linalg.norm(e)), 1e-6)


def build_nodes(masks: Masks, boxes: Boxes, features: Features, graph: Graph, vocab: list[str]) -> list[Node]:
    """The graph's nodes in ID order. vocab is the masks stage's metadata["vocab"]."""
    nodes = []
    for n in np.unique(graph.node):
        rows = np.flatnonzero(graph.node == n)
        votes = np.bincount(masks.prompt[rows], weights=masks.score[rows], minlength=len(vocab))
        lo, hi = fused_box(boxes, rows)
        nodes.append(
            Node(
                id=int(n),
                label=vocab[int(votes.argmax())],
                box_min=lo,
                box_max=hi,
                dino=mean_embedding(features, rows),
                rows=rows,
                frames=np.unique(masks.frame[rows]),
                sam_ids=sorted(set(masks.sam_id[rows].tolist())),
            )
        )
    return nodes


def colours(masks: Masks, node: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """The colours utils/video.py, map_view and graph/plot.py share.

    Returns (n_tracks, 3) uint8 colours by SAM ID, and by node ID the colour of the node's largest track.
    """
    track = np.random.default_rng(0).uniform(90, 255, (int(masks.sam_id.max()) + 1, 3)).astype(np.uint8)
    return track, {int(n): track[np.bincount(masks.sam_id[node == n]).argmax()] for n in np.unique(node)}
