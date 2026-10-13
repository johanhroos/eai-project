"""A run's object map in Rerun: the room as one frozen point cloud, the camera path and the objects as 3D boxes.

The graph's edges (graph/edges.py) are lines between the box centres: world/edges/on in black, from the object on top
to the one under it, and world/edges/near in grey. Hide either in the viewer's entity tree.

Nothing in the room moves. Scrubbing the frame timeline only moves the camera along its path and shows what it saw,
with the objects' 2D boxes; a run of several videos gets one path per video. The point cloud fuses the depth of
FUSED_FRAMES frames spread over the run, lifted with the run's poses (Replica's or VGGT-Omega's). Pixels on depth edges
are dropped, where upsampled depth smears an object into the background behind it, and the rest is thinned to one
point per VOXEL.

In a notebook (the kernel must see the project's packages):

    rr.init("map")
    rr.notebook_show(height=700)
    rerun_map.log_map(footage.make_run("phone/IMG_0644", 1, "sam3", None, "IMG_0644"))

To a file the Rerun viewer opens (rerun <file>, with the same version as rerun-sdk here):

    uv run --no-sync python -m views.rerun_map --source phone/IMG_0644 --stride 1 --vocab data/vocab/IMG_0644.txt
"""

import json

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

from data import footage, replica
from data.stages import Boxes, Graph, Masks, Run, load_stage
from geometry import lift3d
from graph.edges import build_edges
from graph.nodes import build_nodes
from models.sam import mask_boxes

FUSED_FRAMES = 40  # frames whose depth makes up the point cloud
PIXEL_STEP = 2  # every PIXEL_STEP-th pixel in both directions
VOXEL = 0.02  # metres: one point per cube of this size
EDGE_JUMP = 0.05  # drop a pixel when the depth in its 3 x 3 neighbourhood spans more than this share of its depth
IMAGE_SIDE = 640  # long side of the frames shown at the camera
PATH_COLOURS = [(255, 80, 80), (80, 200, 255), (255, 220, 60), (120, 255, 120)]  # camera path of each video


def point_cloud(run: Run) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) float32 world points in metres and their (N, 3) uint8 colours, fused from FUSED_FRAMES frames."""
    frames = footage.frame_numbers(run)
    get = lift3d.geometry_source(run)
    points, colours = [], []
    for i in frames[np.unique(np.linspace(0, len(frames) - 1, FUSED_FRAMES).astype(int))].tolist():
        depth, K, c2w = get(i)
        jump = (maximum_filter(depth, 3) - minimum_filter(depth, 3)) / np.maximum(depth, 1e-6)
        keep = (jump < EDGE_JUMP) & (depth > 0)
        grid = np.zeros_like(keep)
        grid[::PIXEL_STEP, ::PIXEL_STEP] = True
        pts, vu = lift3d.backproject(depth, K, c2w, keep & grid)
        points.append(pts)
        colours.append(footage.load_rgb(run, i)[vu[:, 0], vu[:, 1]])
    points, colours = np.concatenate(points), np.concatenate(colours)
    voxel = np.floor(points / VOXEL).astype(np.int64)
    _, first = np.unique(voxel - voxel.min(0), axis=0, return_index=True)
    return points[first].astype(np.float32), colours[first]


def log_map(run: Run) -> None:
    """Log the run's map to the active Rerun recording (see the module docstring). Needs run.py's cached stages."""
    masks = load_stage(run.stage_dir("masks"), Masks)
    vocab = json.loads((run.stage_dir("masks") / "metadata.json").read_text())["vocab"]
    graph = load_stage(run.stage_dir("graph"), Graph)
    boxes = load_stage(run.stage_dir("boxes"), Boxes)
    nodes = build_nodes(masks, boxes, graph, vocab)
    edges = build_edges(nodes)
    nodes = [n for n in nodes if n.label not in replica.IGNORE_CLASSES]  # walls and floors, as in the evaluation

    # Each object keeps the colour views/video.py gives it: that of its largest SAM track
    track_colour = np.random.default_rng(0).uniform(90, 255, (int(masks.sam_id.max()) + 1, 3)).astype(np.uint8)
    colour = {n.id: track_colour[np.bincount(masks.sam_id[n.rows]).argmax()] for n in nodes}
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="Map"),
                rrb.Spatial2DView(origin="world/camera/image", name="Camera"),
                column_shares=[3, 1],
            )
        )
    )
    infos = [rr.AnnotationInfo(id=n.id + 1, label=f"{n.label} #{n.id}", color=colour[n.id]) for n in nodes]
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world", rr.AnnotationContext(infos), static=True)

    points, colours = point_cloud(run)
    rr.log("world/points", rr.Points3D(points, colors=colours, radii=VOXEL / 2), static=True)
    placed = [n for n in nodes if np.isfinite(n.box_min).all()]
    rr.log(
        "world/objects",
        rr.Boxes3D(
            mins=[n.box_min for n in placed],
            sizes=[n.box_max - n.box_min for n in placed],
            class_ids=[n.id + 1 for n in placed],
            labels=[f"{n.label} #{n.id}" for n in placed],
        ),
        static=True,
    )

    centre = {n.id: (n.box_min + n.box_max) / 2 for n in placed}
    names = {n.id: f"{n.label} #{n.id}" for n in placed}
    for relation, colour, radius in (("on", (20, 20, 20), 0.012), ("near", (150, 150, 150), 0.004)):
        pairs = [(e.a, e.b) for e in edges if e.relation == relation and e.a in centre and e.b in centre]
        if pairs:
            rr.log(
                f"world/edges/{relation}",
                rr.Arrows3D(
                    origins=[centre[a] for a, _ in pairs],
                    vectors=[centre[b] - centre[a] for a, b in pairs],
                    radii=radius,
                    colors=colour,
                    labels=[f"{names[a]} {relation} {names[b]}" for a, b in pairs],
                    show_labels=False,
                )
                if relation == "on"
                else rr.LineStrips3D(
                    [[centre[a], centre[b]] for a, b in pairs],
                    radii=radius,
                    colors=colour,
                    labels=[f"{names[a]} near {names[b]}" for a, b in pairs],
                    show_labels=False,
                ),
                static=True,
            )

    frames = footage.frame_numbers(run).tolist()
    get = lift3d.geometry_source(run)
    cameras = {i: get(i)[1:] for i in frames}  # (K, c2w)
    path = np.array([c2w[:3, 3] for _, c2w in cameras.values()])
    clip = footage.clip_index(run)
    rr.log(
        "world/trajectory",
        rr.LineStrips3D(
            [path[clip == k] for k in np.unique(clip)],  # one line per video
            radii=0.01,
            colors=[PATH_COLOURS[k % len(PATH_COLOURS)] for k in np.unique(clip)],
            labels=[footage.clip_names(run)[k] for k in np.unique(clip)],
        ),
        static=True,
    )

    shown = {n.id for n in nodes}
    xyxy = mask_boxes(masks.rle)
    for i in frames:
        K, c2w = cameras[i]
        rgb = footage.load_rgb(run, i)
        scale = IMAGE_SIDE / max(rgb.shape[:2])
        small = Image.fromarray(rgb).resize((round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)))
        k_small = K.copy()
        k_small[:2] *= scale
        rows = [r for r in np.flatnonzero(masks.frame == i) if int(graph.node[r]) in shown]
        rr.set_time("frame", sequence=i)
        rr.log("world/camera", rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]))
        rr.log(
            "world/camera/image",
            rr.Pinhole(image_from_camera=k_small, width=small.width, height=small.height, image_plane_distance=0.3),
        )
        rr.log("world/camera/image/rgb", rr.Image(np.asarray(small)).compress(jpeg_quality=85))
        rr.log(
            "world/camera/image/objects",
            rr.Boxes2D(
                array=xyxy[rows] * scale,
                array_format=rr.Box2DFormat.XYXY,
                class_ids=[int(graph.node[r]) + 1 for r in rows],
            ),
        )


if __name__ == "__main__":
    import argparse

    from data import vocab

    parser = argparse.ArgumentParser(description="Save a run's map as a Rerun recording, eval/map.rrd.")
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
    out = run.stage_dir("eval") / "map.rrd"
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.init("map_" + run.source.replace("/", "_"))
    rr.save(out)
    log_map(run)
    print(f"Saved {out}")
