"""A picture of a run's graph from above: every node as its 3D box, and the edges between them (graph/edges.py).

Each node is drawn as its box seen from above, in the colour views/video.py gives it and labelled with its label and
ID. "on" edges are arrows from the node on top to the one under it, "near" edges thin dashed lines between the box
centres. Nodes without edges (walls, floors, ceilings and nodes with a box side over edges.MAX_SIDE) are left out, as
in rerun_map. highlight draws chosen nodes thicker and dims the rest, such as the answer of a --near query.

In a notebook:      top_down.plot(run)
                    top_down.plot(run, highlight=[37, 82])
From the shell:     uv run --no-sync python -m views.top_down --source replica/room_0/00   (saves eval/edges.png)
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, patheffects

from data.stages import Boxes, Graph, Masks, Run, load_stage
from graph.edges import MAX_SIDE, STRUCTURE, build_edges
from graph.nodes import build_nodes, colours
from views import rerun_map

MARGIN = 0.5  # metres of room shown around the objects


def load_graph(run: Run) -> tuple[list, list]:
    """The run's nodes (graph.nodes.build_nodes) and edges (graph.edges.build_edges). Needs run.py's cached stages."""
    masks = load_stage(run.stage_dir("masks"), Masks)
    vocab = json.loads((run.stage_dir("masks") / "metadata.json").read_text())["vocab"]
    graph = load_stage(run.stage_dir("graph"), Graph)
    boxes = load_stage(run.stage_dir("boxes"), Boxes)
    nodes = build_nodes(masks, boxes, graph, vocab)
    return nodes, build_edges(nodes)


def plot(run: Run, highlight: list[int] | None = None, near: bool = True) -> plt.Figure:
    """The view from above of the run's nodes and edges (see the module docstring). near=False leaves out "near"."""
    masks = load_stage(run.stage_dir("masks"), Masks)
    nodes, edges = load_graph(run)
    _, node_colour = colours(masks, load_stage(run.stage_dir("graph"), Graph).node)
    shown = [
        n
        for n in nodes
        if n.label not in STRUCTURE and np.isfinite(n.box_min).all() and (n.box_max - n.box_min).max() <= MAX_SIDE
    ]
    lit = set(highlight or [])

    def alpha(*ids):
        return 1.0 if not lit or lit & set(ids) else 0.15

    fig, ax = plt.subplots(figsize=(13, 10))
    points, rgb = rerun_map.point_cloud(run)
    corners = [c for n in shown for c in (n.box_min[:2], n.box_max[:2])] or points[:, :2]  # no nodes: the whole room
    lo, hi = np.min(corners, axis=0) - MARGIN, np.max(corners, axis=0) + MARGIN
    inside = ((points[:, :2] >= lo) & (points[:, :2] <= hi)).all(axis=1)
    ax.scatter(points[inside, 0], points[inside, 1], c=rgb[inside] / 255, s=1, alpha=0.45, linewidths=0)

    centre = {n.id: (n.box_min[:2] + n.box_max[:2]) / 2 for n in shown}
    for e in edges:
        if e.a not in centre or e.b not in centre or (e.relation == "near" and not near):
            continue
        (x0, y0), (x1, y1) = centre[e.a], centre[e.b]
        if e.relation == "on":
            arrow = patches.FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color="k", lw=1.6, alpha=alpha(e.a, e.b)
            )
            ax.add_patch(arrow)
        else:
            ax.plot([x0, x1], [y0, y1], color="0.4", ls="--", lw=0.8, alpha=alpha(e.a, e.b) * 0.8)

    for n in shown:
        (x0, y0), (x1, y1) = n.box_min[:2], n.box_max[:2]
        colour = node_colour[n.id] / 255
        bold = n.id in lit
        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=True,
            facecolor=(*colour, 0.25 * alpha(n.id)),
            edgecolor=(*colour * 0.7, alpha(n.id)),
            lw=3 if bold else 1.2,
        )
        ax.add_patch(rect)
        x, y = centre[n.id]
        text = ax.text(
            x, y, f"{n.label} {n.id}", fontsize=10 if bold else 7.5, weight="bold" if bold else "normal",
            ha="center", va="center", alpha=alpha(n.id), zorder=4,
        )  # fmt: skip
        text.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground="white")])

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    n_on = sum(e.relation == "on" for e in edges)
    ax.set_title(
        f"{run.source}: {len(shown)} of {len(nodes)} nodes from above, as their 3D boxes. "
        f"Arrows: {n_on} 'on' edges, from the object on top to the one under it."
        + (f"\nDashed: {len(edges) - n_on} 'near' edges, boxes at most 25 cm apart." if near else ""),
        loc="left",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse

    from data import footage, vocab

    parser = argparse.ArgumentParser(description="Save a picture of a run's graph nodes and edges as eval/edges.png.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--mask-source", default="sam3", choices=["sam3", "gt"])
    parser.add_argument("--pose-source", choices=["gt", "vggt"], help="as for run.py")
    parser.add_argument("--vocab", help="as for run.py")
    parser.add_argument("--highlight", type=int, nargs="*", help="node IDs to draw thicker")
    parser.add_argument("--no-near", action="store_true", help="leave out the 'near' edges")
    args = parser.parse_args()
    try:
        run = footage.make_run(args.source, args.stride, args.mask_source, args.pose_source, vocab.tag(args.vocab))
    except ValueError as e:
        parser.error(str(e))
    plt.switch_backend("Agg")  # no display here
    out = run.stage_dir("eval") / "edges.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(run, args.highlight, near=not args.no_near).savefig(out, dpi=110)
    print(f"Saved {out}")
