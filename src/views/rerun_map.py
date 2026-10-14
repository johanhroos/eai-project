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

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

from data import footage, replica
from data.stages import Run, bounding_boxes
from evaluation import ground_truth
from geometry.depth import backproject, geometry_source
from graph.edges import build_edges
from graph.nodes import colours, load_nodes

FUSED_FRAMES = 40  # frames whose depth makes up the point cloud
PIXEL_STEP = 2  # every PIXEL_STEP-th pixel in both directions
VOXEL = 0.02  # metres: one point per cube of this size
EDGE_JUMP = 0.05  # drop a pixel when the depth in its 3 x 3 neighbourhood spans more than this share of its depth
IMAGE_SIDE = 640  # long side of the frames shown at the camera
PATH_COLOURS = [(255, 80, 80), (80, 200, 255), (255, 220, 60), (120, 255, 120)]  # camera path of each video
GT_COLOUR = (230, 230, 230)  # Replica's GT boxes (log_gt_boxes)


def point_cloud(run: Run) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) float32 world points in metres and their (N, 3) uint8 colours, fused from FUSED_FRAMES frames."""
    frames = footage.frame_numbers(run)
    get = geometry_source(run)
    points, rgb = [], []
    for i in frames[np.unique(np.linspace(0, len(frames) - 1, FUSED_FRAMES).astype(int))].tolist():
        depth, K, c2w = get(i)
        jump = (maximum_filter(depth, 3) - minimum_filter(depth, 3)) / np.maximum(depth, 1e-6)
        keep = (jump < EDGE_JUMP) & (depth > 0)
        grid = np.zeros_like(keep)
        grid[::PIXEL_STEP, ::PIXEL_STEP] = True
        pts, vu = backproject(depth, K, c2w, keep & grid)
        points.append(pts)
        rgb.append(footage.load_rgb(run, i)[vu[:, 0], vu[:, 1]])
    points, rgb = np.concatenate(points), np.concatenate(rgb)
    voxel = np.floor(points / VOXEL).astype(np.int64)
    _, first = np.unique(voxel - voxel.min(0), axis=0, return_index=True)
    return points[first].astype(np.float32), rgb[first]


def log_map(run: Run) -> None:
    """Log the run's map to the active Rerun recording (see the module docstring). Needs run.py's cached stages."""
    masks, graph, nodes = load_nodes(run)
    edges = build_edges(nodes)
    nodes = [n for n in nodes if n.label not in replica.IGNORE_CLASSES]  # walls and floors, as in the evaluation

    # Each object keeps the colour views/video.py gives it: that of its largest SAM track
    _, node_colour = colours(masks, graph.node)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="Map"),
                rrb.Spatial2DView(origin="world/camera/image", name="Camera"),
                column_shares=[3, 1],
            )
        )
    )
    infos = [rr.AnnotationInfo(id=n.id + 1, label=f"{n.label} #{n.id}", color=node_colour[n.id]) for n in nodes]
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world", rr.AnnotationContext(infos), static=True)

    points, rgb = point_cloud(run)
    rr.log("world/points", rr.Points3D(points, colors=rgb, radii=VOXEL / 2), static=True)
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
    get = geometry_source(run)
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
    xyxy = bounding_boxes(masks.rle)
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


def log_gt_boxes(run: Run) -> None:
    """Log a Replica run's GT boxes (ground_truth.boxes_in_run) under world/gt, to compare with the map's boxes.

    They are light grey, with the class on hover; hide them in the viewer's entity tree. With VGGT-Omega's poses they
    are moved into the map's frame by the camera path's best fit, so they are only as well placed as that fit.
    """
    from scipy.spatial.transform import Rotation

    centre, half, rotation, names = ground_truth.boxes_in_run(run)
    turn = rr.Quaternion(xyzw=Rotation.from_matrix(rotation).as_quat())
    rr.log(
        "world/gt",
        rr.Boxes3D(
            centers=centre,
            half_sizes=half,
            quaternions=[turn] * len(centre),
            colors=GT_COLOUR,
            radii=0.004,
            labels=[f"GT {name}" for name in names],
            show_labels=False,
        ),
        static=True,
    )


if __name__ == "__main__":
    import argparse

    from data import cli

    parser = argparse.ArgumentParser(description="Save a run's map as a Rerun recording, eval/map.rrd.")
    cli.add_run_arguments(parser, source=None)
    parser.add_argument("--gt", action="store_true", help="also log the GT boxes (Replica only)")
    args = parser.parse_args()
    run = cli.runs(parser, args)[0]
    out = run.stage_dir("eval") / "map.rrd"
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.init("map_" + run.source.replace("/", "_"))
    rr.save(out)
    log_map(run)
    if args.gt:
        log_gt_boxes(run)
    print(f"Saved {out}")
