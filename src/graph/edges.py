"""Edges between object nodes, from their 3D boxes, the queries over them and the nearest-object query.

  on    node a rests on or in node b: most of a's footprint lies over b's, a's bottom is above b's bottom, at most
        ON_GAP above b's top and at most ON_DEPTH below it, and a is the smaller one. A cushion's box lies inside the
        sofa's, since the sofa's box includes its back, so a's bottom touching b's top would miss it.
  near  the boxes are at most NEAR metres apart (0 when they touch or overlap), and neither rests on the other

Walls, floors and ceilings (STRUCTURE) get no edges: they touch almost everything. Neither do nodes with a box side
over MAX_SIDE, such as a rug, or a node whose box spans the room because a SAM mask spilled onto the background.
The nodes' boxes are world frame with z up (Replica's, or VGGT-Omega's after models/vggt.py).

Edges are cheap to compute from the nodes, so like graph/nodes.py they aren't a stage of their own.

From the shell, after run.py has cached the stages:

    uv run --no-sync python -m graph.edges --source replica/room_0/00
"""

from dataclasses import dataclass

import numpy as np

from graph.nodes import Node

NEAR = 0.25  # metres between two boxes up to which their nodes get a "near" edge
ON_GAP = 0.1  # metres a's bottom may be above b's top for "a on b"
ON_RISE = 0.05  # metres a's bottom must be above b's bottom: a box inside b, not standing next to it on the floor
ON_DEPTH = 0.75  # metres a's bottom may be below b's top: an armchair's seat is about 0.7 m below its back
ON_COVER = 0.5  # share of a's footprint over b's for "a on b"
MAX_SIDE = 3.0  # metres; a node with a longer box side gets no edges
STRUCTURE = {"wall", "floor", "ceiling"}


@dataclass
class Edge:
    a: int  # node ID; for "on", the node resting on b
    b: int  # node ID
    relation: str  # "on" or "near"
    gap: float  # metres between the two boxes, 0 when they touch or overlap


def box_corners(nodes: list[Node]) -> tuple[np.ndarray, np.ndarray]:
    """(M, 3) box_min and box_max of the nodes, in list order."""
    if not nodes:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.stack([n.box_min for n in nodes]), np.stack([n.box_max for n in nodes])


def box_gap(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """(M, M) distance between every pair of boxes, 0 where they touch or overlap. NaN where a box is NaN."""
    apart = np.maximum(lo[:, None] - hi[None], lo[None] - hi[:, None])  # per axis, negative where they overlap
    return np.linalg.norm(np.clip(apart, 0, None), axis=2)


def rests_on(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """(M, M) bool, [a, b] where box a rests on or in box b (see the module docstring)."""
    footprint = np.prod(
        np.clip(np.minimum(hi[:, None, :2], hi[None, :, :2]) - np.maximum(lo[:, None, :2], lo[None, :, :2]), 0, None),
        axis=2,
    )
    cover = footprint / np.maximum(np.prod(hi[:, :2] - lo[:, :2], axis=1), 1e-6)[:, None]
    bottom, top = lo[:, 2], hi[:, 2]
    volume = np.prod(hi - lo, axis=1)
    on = (cover >= ON_COVER) & (bottom[:, None] >= bottom[None] + ON_RISE) & (bottom[:, None] <= top[None] + ON_GAP)
    on &= bottom[:, None] >= top[None] - ON_DEPTH
    on &= volume[:, None] < volume[None]
    np.fill_diagonal(on, False)
    return on


def gets_edges(nodes: list[Node]) -> np.ndarray:
    """(M,) bool, whether each node can have edges: not with a STRUCTURE label, no box or a box side over MAX_SIDE."""
    lo, hi = box_corners(nodes)
    structure = np.array([n.label in STRUCTURE for n in nodes], bool)
    return ~structure & np.isfinite(lo).all(1) & ((hi - lo).max(1) <= MAX_SIDE)


def build_edges(nodes: list[Node]) -> list[Edge]:
    """The edges between the nodes, from build_nodes.

    Nodes without a box, with a STRUCTURE label or with a box side over MAX_SIDE get none (gets_edges).
    """
    if not nodes:
        return []
    lo, hi = box_corners(nodes)
    gap = box_gap(lo, hi)
    on = rests_on(lo, hi)
    keep = gets_edges(nodes)
    edges = []
    for i, j in zip(*np.triu_indices(len(nodes), 1), strict=True):
        if not (keep[i] and keep[j]) or not (on[i, j] or on[j, i] or gap[i, j] <= NEAR):
            continue
        a, b = (i, j) if on[i, j] or not on[j, i] else (j, i)
        relation = "on" if on[a, b] else "near"
        edges.append(Edge(nodes[a].id, nodes[b].id, relation, float(gap[i, j])))
    return edges


def neighbours(edges: list[Edge], node: int) -> list[tuple[str, int]]:
    """The edges of a node as (relation seen from it, other node ID).

    The relation is "on" (it rests on the other node), "under" (the other rests on it) or "near".
    """
    out = []
    for e in edges:
        if e.a == node:
            out.append((e.relation, e.b))
        elif e.b == node:
            out.append(("under" if e.relation == "on" else "near", e.a))
    return out


def related(edges: list[Edge], anchors: list[int], relation: str) -> list[tuple[int, str, int, float]]:
    """The nodes with an edge to one of the anchor node IDs, as (ID, relation seen from it, anchor ID, box gap).

    relation "on" keeps the nodes that rest on an anchor. "near" keeps every edge, "on" and "under" too: a cushion on a
    sofa has an "on" edge to it and no "near" one, but it is near it. A node with edges to several anchors appears
    once, for the closest. The anchors themselves can appear, when two of them have an edge. Closest first.
    """
    best = {}
    for e in edges:
        for node, anchor, rel in ((e.a, e.b, e.relation), (e.b, e.a, "under" if e.relation == "on" else "near")):
            if anchor not in anchors or (relation == "on" and rel != "on"):
                continue
            if node not in best or e.gap < best[node][3]:
                best[node] = (node, rel, anchor, e.gap)
    return sorted(best.values(), key=lambda r: (r[3], r[0]))


def nearest(nodes: list[Node], anchor: int, candidates: list[int]) -> list[tuple[int, float, float]]:
    """The candidate node IDs from nearest to farthest from the anchor node, as (ID, box gap, centre distance).

    Nearest is the smallest gap between the boxes, then the closest centres: the cushions on a sofa all touch it, and
    the one in the middle of it comes first. The anchor itself and nodes without a box are left out. nodes is the
    list from build_nodes, whose IDs are its indices.
    """
    return nearest_boxes(*box_corners(nodes), anchor, candidates)


def nearest_boxes(lo: np.ndarray, hi: np.ndarray, anchor: int, candidates: list[int]) -> list[tuple[int, float, float]]:
    """Like nearest, for any (M, 3) boxes such as the GT objects' (metrics.nearest_objects): rows, not node IDs."""
    a_lo, a_hi = lo[anchor : anchor + 1], hi[anchor : anchor + 1]
    gap = np.linalg.norm(np.clip(np.maximum(lo - a_hi, a_lo - hi), 0, None), axis=1)  # as box_gap, for one row
    centre = np.linalg.norm((lo + hi) / 2 - (a_lo + a_hi) / 2, axis=1)
    rows = [c for c in candidates if c != anchor and np.isfinite(gap[c])]
    rows.sort(key=lambda c: (round(float(gap[c]), 2), float(centre[c])))
    return [(c, float(gap[c]), float(centre[c])) for c in rows]


if __name__ == "__main__":
    import argparse
    from collections import Counter

    from data import cli
    from graph.nodes import load_nodes

    parser = argparse.ArgumentParser(description="Print the edges of a run's graph.")
    cli.add_run_arguments(parser)
    args = parser.parse_args()
    run = cli.runs(parser, args)[0]
    _, _, nodes = load_nodes(run)
    edges = build_edges(nodes)
    counts = Counter(e.relation for e in edges)
    print(f"{run.source}: {len(nodes)} nodes, {len(edges)} edges ({counts['on']} on, {counts['near']} near)")
    for e in sorted(edges, key=lambda e: (e.relation != "on", e.a, e.b)):
        a, b = nodes[e.a], nodes[e.b]
        print(f"  {a.label:>14} {a.id:<3} {e.relation:<4} {b.label} {b.id}  ({e.gap:.2f} m apart)")
