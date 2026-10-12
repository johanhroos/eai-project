"""A 2D picture of a run's graph, to see why SAM tracks were or weren't merged into one object.

  timeline  one row per SAM track over the run's frames, grouped and coloured by node. In a node, tracks that overlap in
            time are duplicates, tracks that follow each other are re-entries, and tracks on both sides of a video
            boundary link two videos.
  above     the room's point cloud from above, with every track as a dot at its 3D box centre, coloured and numbered
            by node. Solid lines are the graph's merges; dashed lines are near-misses, pairs of nodes whose boxes touch
            or that have the same label within NEAR metres, except pairs seen apart in the same frames. A text panel
            lists each merge with its rule and value, and each near-miss with what kept the two apart.

The merges come from rebuilding the graph with graph.build(trace=...), which must give the cached graph stage.

In a notebook:      plot.plot(run)
From the shell:     uv run --no-sync python -m graph.plot --source phone/IMG_0644 --stride 3   (saves eval/graph.png)
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

from graph import graph
from graph.node import colours, fused_box
from utils import footage, map_view, replica
from utils.schema import Boxes, Features, Graph, Masks, Run, load_stage

NEAR = 1.0  # metres between same-label nodes that still count as a near-miss
MARGIN = 0.5  # metres of room shown around the objects
MAX_LINES = 70  # of the text panel


def plot(run: Run) -> plt.Figure:
    """The timeline and the view from above of the run's graph (see the module docstring). Needs run.py's stages."""
    masks = load_stage(run.stage_dir("masks"), Masks)
    vocab = json.loads((run.stage_dir("masks") / "metadata.json").read_text())["vocab"]
    boxes = load_stage(run.stage_dir("boxes"), Boxes)
    trace = []
    node = graph.build(masks, boxes, load_stage(run.stage_dir("features"), Features), trace).node
    if not np.array_equal(node, load_stage(run.stage_dir("graph"), Graph).node):
        raise ValueError(f"{run.stage_dir('graph')} was built with other parameters: rerun run.py first")
    _, node_colour = colours(masks, node)

    def label(rows):  # the prompt with the most SAM score, as graph and node do
        return vocab[int(np.bincount(masks.prompt[rows], weights=masks.score[rows]).argmax())]

    tracks = np.unique(masks.sam_id)
    rows_of = {int(t): np.flatnonzero(masks.sam_id == t) for t in tracks}
    node_of = {t: int(node[r[0]]) for t, r in rows_of.items()}
    nodes = np.unique(node)
    node_rows = {int(n): np.flatnonzero(node == n) for n in nodes}
    order = sorted(rows_of, key=lambda t: (node_of[t], masks.frame[rows_of[t]].min()))

    height = 1 + 0.22 * len(order)
    fig = plt.figure(figsize=(16, height + 10), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[height, 10])
    _timeline(top.subplots(), run, masks, order, rows_of, node_of, node_colour, label)
    left, right = bottom.subplots(1, 2, width_ratios=[3, 2])
    lines = _above(left, run, masks, boxes, trace, rows_of, node_of, node_rows, node_colour, label)
    right.axis("off")
    if len(lines) > MAX_LINES:
        lines = [*lines[: MAX_LINES - 1], f"  ... and {len(lines) - MAX_LINES + 1} more"]
    right.text(0, 1, "\n".join(lines), family="monospace", fontsize=7, va="top")
    fig.suptitle(f"{run.source}: {len(rows_of)} SAM tracks, {len(nodes)} objects")
    return fig


def _timeline(ax, run, masks, order, rows_of, node_of, node_colour, label):
    numbers, clip = footage.frame_numbers(run), footage.clip_index(run)
    position = {int(f): k for k, f in enumerate(numbers)}  # videos side by side, without gaps
    for y, t in enumerate(order):
        seen = np.array(sorted(position[int(f)] for f in masks.frame[rows_of[t]]))
        runs = np.split(seen, np.flatnonzero(np.diff(seen) > 1) + 1)  # stretches of consecutive frames
        colour = node_colour[node_of[t]] / 255
        ax.broken_barh([(r[0], r[-1] - r[0] + 1) for r in runs], (y - 0.4, 0.8), color=colour)
    ax.set_yticks(range(len(order)), [f"node {node_of[t]}: {label(rows_of[t])} (track {t})" for t in order], fontsize=8)
    ax.invert_yaxis()
    for y in np.flatnonzero(np.diff([node_of[t] for t in order])):  # a line between nodes
        ax.axhline(y + 0.5, color="0.85", lw=0.8)
    names = footage.clip_names(run)
    for k in np.unique(clip):
        start = int(np.flatnonzero(clip == k)[0])
        if k:
            ax.axvline(start - 0.5, color="k", ls="--", lw=1)
        ax.text(start, -0.8, names[k], fontsize=9, va="bottom")
    ax.set_xlim(-0.5, len(numbers) - 0.5)
    ax.set_xlabel("frame of the run (every stride-th frame, videos one after the other)")
    ax.set_title("When each SAM track is seen, grouped by object", loc="left")


def _above(ax, run, masks, boxes, trace, rows_of, node_of, node_rows, node_colour, label) -> list[str]:
    """Draw the view from above on ax and return the lines of the text panel: the merges and the near-misses."""
    centre = {}
    for t, rows in rows_of.items():
        lo, hi = fused_box(boxes, rows)
        if np.isfinite(lo).all():
            centre[t] = (lo + hi)[:2] / 2
    lo_hi = {n: fused_box(boxes, rows) for n, rows in node_rows.items()}
    node_centre = {n: (lo + hi)[:2] / 2 for n, (lo, hi) in lo_hi.items() if np.isfinite(lo).all()}
    xy = np.array(list(centre.values()))
    view_lo, view_hi = xy.min(0) - MARGIN, xy.max(0) + MARGIN  # zoom on the objects, not the whole room
    points, rgb = map_view.point_cloud(run)
    shown = ((points[:, :2] >= view_lo) & (points[:, :2] <= view_hi)).all(axis=1)
    ax.scatter(points[shown, 0], points[shown, 1], c=rgb[shown] / 255, s=1, alpha=0.3, linewidths=0)
    names = {n: label(rows) for n, rows in node_rows.items()}
    lines = ["MERGES (rule, value)"]
    for rule, a, b, value in trace:
        lines.append(f"  {node_of[a]:>3} {names[node_of[a]]:<10} track {a:>3} + {b:<3}  {rule} {value:.2f}")
        if a in centre and b in centre:
            (x0, y0), (x1, y1) = centre[a], centre[b]
            ax.plot([x0, x1], [y0, y1], color=node_colour[node_of[a]] / 255 * 0.7, lw=2.5)

    # Near-misses between nodes, with what kept them apart
    shared = graph.overlap_share(masks, np.unique(masks.sam_id, return_inverse=True)[1])
    tracks = np.unique(masks.sam_id)
    apart = {(int(tracks[a]), int(tracks[b])) for (a, b), share in shared.items() if share <= graph.APART_SHARE}
    ids = sorted(node_centre)
    lo, hi = np.array([lo_hi[n][0] for n in ids]), np.array([lo_hi[n][1] for n in ids])
    iou, inside = graph.padded_box_iou(lo, hi), graph.padded_box_containment(lo, hi)
    inside = np.maximum(inside, inside.T)
    members = {n: [t for t in rows_of if node_of[t] == n] for n in ids}
    misses = []  # (seen apart, line)
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            b = ids[j]
            if names[a] in replica.IGNORE_CLASSES or names[b] in replica.IGNORE_CLASSES:
                continue  # walls and floors touch everything
            distance = float(np.linalg.norm(node_centre[a] - node_centre[b]))
            if not (iou[i, j] > 0 or (names[a] == names[b] and distance < NEAR)):
                continue
            (x0, y0), (x1, y1) = node_centre[a], node_centre[b]
            seen_apart = any((min(s, t), max(s, t)) in apart for s in members[a] for t in members[b])
            if seen_apart:
                why = "seen apart in the same frames"
            elif names[a] != names[b]:
                why = "different labels"
            else:
                why = f"IoU {iou[i, j]:.2f} < {graph.BOX_IOU:g}, inside {inside[i, j]:.2f} < {graph.CONTAIN:g}"
            if not seen_apart:  # the usual case, so only in the text panel
                ax.plot([x0, x1], [y0, y1], color="0.35", ls="--", lw=0.9)
            misses.append((seen_apart, f"  {a:>3} {names[a]:<10} / {b:>3} {names[b]:<10} {distance:4.2f} m  {why}"))
    misses.sort(key=lambda m: m[0])  # the ones the graph could have merged first
    lines += ["", f"NEAR-MISSES (boxes touch, or same label within {NEAR:g} m; no walls, floors, ceilings)"]
    lines += [line for _, line in misses]
    for t, (x, y) in centre.items():
        ax.scatter(x, y, color=node_colour[node_of[t]] / 255, s=40, edgecolors="k", linewidths=0.6, zorder=3)
    for n, (x, y) in node_centre.items():
        text = ax.text(x, y, f" {n}", fontsize=9, weight="bold", va="bottom", zorder=4)
        text.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground="white")])
    ax.set_xlim(view_lo[0], view_hi[0])
    ax.set_ylim(view_lo[1], view_hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    missing = len(rows_of) - len(centre)
    ax.set_title(
        "From above: SAM tracks (dots) at their 3D box centres, coloured and numbered by object.\n"
        "Solid: merges. Dashed: near-misses the graph could have merged. Details on the right"
        + (f". {missing} tracks have no 3D box." if missing else "."),
        loc="left",
        fontsize=9,
    )
    return lines


if __name__ == "__main__":
    import argparse

    from utils import vocab

    parser = argparse.ArgumentParser(description="Save a picture of a run's graph as eval/graph.png.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--mask-source", default="sam3", choices=["sam3", "gt"])
    parser.add_argument("--pose-source", choices=["gt", "vggt"], help="as for run.py")
    parser.add_argument("--vocab", help="as for run.py")
    args = parser.parse_args()
    try:
        run = footage.make_run(args.source, args.stride, args.mask_source, args.pose_source, vocab.tag(args.vocab))
    except ValueError as e:
        parser.error(str(e))
    plt.switch_backend("Agg")  # no display here
    out = run.stage_dir("eval") / "graph.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(run).savefig(out, dpi=110)
    print(f"Saved {out}")
