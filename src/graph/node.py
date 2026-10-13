"""Object nodes of the scene graph, put together from the graph stage's node ID per mask."""

from dataclasses import dataclass

import numpy as np

from utils.lift3d import reliable
from utils.schema import Boxes, Graph, Masks


@dataclass
class Node:
    id: int
    label: str  # the graph's label for it (graph/refine.py), else the prompt with the most SAM score over its masks
    box_min: np.ndarray  # (3,) float32, world frame, metres; NaN if none of the masks has a box
    box_max: np.ndarray  # (3,) float32
    rows: np.ndarray  # (k,) int, the node's rows in the masks stage
    frames: np.ndarray  # (f,) int32, the frames it is seen in, sorted
    sam_ids: list[int]  # the SAM tracks whose masks it holds


BOX_PERCENTILES = (2, 98)  # an object's box leaves out this share of its pooled points on each side


def fit_box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The box around (k, 3) points without their outer BOX_PERCENTILES.

    The bounds are sample values (inverted CDF), so for fewer than 50 points, such as a few boxes' corners, they are
    the plain minimum and maximum.
    """
    lo, hi = np.percentile(points, BOX_PERCENTILES, axis=0, method="inverted_cdf")
    return lo.astype(np.float32), hi.astype(np.float32)


def mask_points(boxes: Boxes, rows: np.ndarray) -> np.ndarray:
    """(k, 3) float32: the masks' stored point samples, or their boxes' corners when the boxes stage has no points."""
    if boxes.points is None:
        return np.concatenate([boxes.box_min[rows], boxes.box_max[rows]]).astype(np.float32)
    points = boxes.points[rows].reshape(-1, 3).astype(np.float32)
    return points[np.isfinite(points).all(axis=1)]


def most_confident(boxes: Boxes, rows: np.ndarray) -> int | None:
    """The row with a box whose depth is most confident (the first on ties), or None if no row has a box."""
    rows = rows[boxes.n_points[rows] > 0]
    if not len(rows):
        return None
    if boxes.conf is None:
        return int(rows[0])
    return int(rows[np.argmax(np.nan_to_num(boxes.conf[rows], nan=-np.inf))])


def node_box(boxes: Boxes, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An object's box from its masks: around the pooled points of the reliable ones (lift3d.reliable).

    The views each see part of the object, so their pooled points cover the space it fills, also behind occluders.
    Pooling rather than the union of the masks' boxes keeps one view's stray depth from setting the box. Without a
    reliable mask, the box of the mask with the most confident depth; NaN if none of the masks has a box.
    graph.association.ObjectMemory keeps its nodes' boxes by the same rule. A saved graph has better ones, from every
    view of each node (graph/shape.py).
    """
    good = rows[reliable(boxes)[rows]]
    if len(good):
        return fit_box(mask_points(boxes, good))
    best = most_confident(boxes, rows)
    if best is None:
        return np.full(3, np.nan, np.float32), np.full(3, np.nan, np.float32)
    return boxes.box_min[best].astype(np.float32), boxes.box_max[best].astype(np.float32)


def build_nodes(masks: Masks, boxes: Boxes, graph: Graph, vocab: list[str]) -> list[Node]:
    """The graph's nodes in ID order. vocab is the masks stage's metadata["vocab"]. Rows with node -1 are left out.

    The boxes and labels are the graph's own (graph/shape.py, graph/refine.py) where it has them, else node_box's
    and the prompt with the most SAM score.
    """
    nodes = []
    for n in np.unique(graph.node[graph.node >= 0]):
        rows = np.flatnonzero(graph.node == n)
        if graph.label is not None:
            label = int(graph.label[n])
        else:
            label = int(np.bincount(masks.prompt[rows], weights=masks.score[rows]).argmax())
        if graph.box_min is not None:
            lo, hi = graph.box_min[n].copy(), graph.box_max[n].copy()
        else:
            lo, hi = node_box(boxes, rows)
        nodes.append(
            Node(
                id=int(n),
                label=vocab[label],
                box_min=lo,
                box_max=hi,
                rows=rows,
                frames=np.unique(masks.frame[rows]),
                sam_ids=sorted(set(masks.sam_id[rows].tolist())),
            )
        )
    return nodes


def colours(masks: Masks, node: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """The colours utils/video.py, map_view and graph/plot_edges.py share.

    Returns (n_tracks, 3) uint8 colours by SAM ID, and by node ID the colour of the node's largest track.
    """
    track = np.random.default_rng(0).uniform(90, 255, (int(masks.sam_id.max()) + 1, 3)).astype(np.uint8)
    return track, {int(n): track[np.bincount(masks.sam_id[node == n]).argmax()] for n in np.unique(node[node >= 0])}
