"""A picture of a run's graph from above: every node as its 3D box, and the edges between them (graph/edges.py).

Each node is drawn as its box seen from above, in the colour views/video.py gives it and labelled with its label and
ID. "on" edges are arrows from the node on top to the one under it, "near" edges thin dashed lines between the box
centres. Nodes without edges (walls, floors, ceilings and nodes with a box side over edges.MAX_SIDE) are left out, as
in rerun_map. highlight draws chosen nodes thicker and dims the rest, such as the answer of a --nearest query. On Replica,
gt=True also draws the GT objects' boxes (ground_truth.boxes_in_run) as dashed outlines, to compare the boxes with.

In a notebook:      top_down.plot(run)
                    top_down.plot(run, highlight=[37, 82])
                    top_down.plot(run, gt=True)
From the shell:     uv run --no-sync python -m views.top_down --source replica/room_0/00   (saves eval/edges.png)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, patheffects

from data.stages import Run
from evaluation import ground_truth
from graph.edges import MAX_SIDE, STRUCTURE, build_edges
from graph.nodes import colours, load_nodes
from views import rerun_map

MARGIN = 0.5  # metres of room shown around the objects


def plot(run: Run, highlight: list[int] | None = None, near: bool = True, gt: bool = False) -> plt.Figure:
    """The view from above of the run's nodes and edges (see the module docstring). near=False leaves out "near"."""
    masks, graph, nodes = load_nodes(run)
    edges = build_edges(nodes)
    _, node_colour = colours(masks, graph.node)
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

    if gt:
        centre, half, rotation, _ = ground_truth.boxes_in_run(run)
        square = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]])  # the bottom face, in order
        for c, h in zip(centre, half, strict=True):
            footprint = (c + (square * h) @ rotation.T)[:, :2]
            ax.add_patch(patches.Polygon(footprint, closed=True, fill=False, ls="--", lw=0.9, ec="0.15", zorder=3))

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

    from data import cli

    parser = argparse.ArgumentParser(description="Save a picture of a run's graph nodes and edges as eval/edges.png.")
    cli.add_run_arguments(parser, source=None)
    parser.add_argument("--highlight", type=int, nargs="*", help="node IDs to draw thicker")
    parser.add_argument("--no-near", action="store_true", help="leave out the 'near' edges")
    parser.add_argument("--gt", action="store_true", help="also draw the GT boxes (Replica only)")
    args = parser.parse_args()
    run = cli.runs(parser, args)[0]
    plt.switch_backend("Agg")  # no display here
    out = run.stage_dir("eval") / "edges.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(run, args.highlight, near=not args.no_near, gt=args.gt).savefig(out, dpi=110)
    print(f"Saved {out}")
