"""Object nodes of the scene graph, put together from the graph stage's node ID per mask."""

from dataclasses import dataclass

import numpy as np

from utils.schema import Boxes, Graph, Masks


@dataclass
class Node:
    id: int
    label: str  # the prompt with the most SAM score summed over the node's masks
    box_min: np.ndarray  # (3,) float32, world frame, metres; NaN if none of the masks has a box
    box_max: np.ndarray  # (3,) float32
    rows: np.ndarray  # (k,) int, the node's rows in the masks stage
    frames: np.ndarray  # (f,) int32, the frames it is seen in, sorted
    sam_ids: list[int]  # the SAM tracks whose masks it holds


def union_box(boxes: Boxes, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The box around every box of the masks in rows: the smallest minimum and the largest maximum per axis.

    An object's views each see part of it, so the union is the space it fills, also behind occluders. Each mask's box
    already leaves out its stray depth points (lift3d.mask_to_box). NaN if none of the masks has a box.
    """
    rows = rows[boxes.n_points[rows] > 0]
    if not len(rows):
        return np.full(3, np.nan, np.float32), np.full(3, np.nan, np.float32)
    return boxes.box_min[rows].min(0).astype(np.float32), boxes.box_max[rows].max(0).astype(np.float32)


def build_nodes(masks: Masks, boxes: Boxes, graph: Graph, vocab: list[str]) -> list[Node]:
    """The graph's nodes in ID order. vocab is the masks stage's metadata["vocab"]. Rows with node -1 are left out."""
    nodes = []
    for n in np.unique(graph.node[graph.node >= 0]):
        rows = np.flatnonzero(graph.node == n)
        votes = np.bincount(masks.prompt[rows], weights=masks.score[rows], minlength=len(vocab))
        lo, hi = union_box(boxes, rows)
        nodes.append(
            Node(
                id=int(n),
                label=vocab[int(votes.argmax())],
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
