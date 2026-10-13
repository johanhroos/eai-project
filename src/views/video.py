"""Render runs to MP4: every frame with an outlined mask, label and ID per object, one panel per ID source side by side.

  tracks  SAM tracks of the masks stage: label = prompt, ID = SAM track
  graph   nodes of the graph stage: a node's masks in a frame are drawn as one, their union, label = node label
  gt      Replica ground truth: label = class, ID = Replica object, without masks under ground_truth.MIN_PIXELS

Like the evaluation, every panel leaves out walls, floors, ceilings and unlabelled objects (replica.IGNORE_CLASSES)
unless --show-ignored is given.

An ID keeps its colour for the whole video, and a node takes the colour of its largest SAM track: a duplicate that the
graph merges has two colours in the tracks panel and one in the graph panel. Reads the cached stages, so run run.py with
the same arguments first. Writes eval/<panels>.mp4 in each run's output directory.
"""

import argparse
import json
from dataclasses import replace

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion
from torchcodec.encoders import VideoEncoder

from data import footage, replica, vocab
from data.stages import Boxes, Graph, Masks, Run, load_stage, rle_area
from evaluation import ground_truth
from graph.nodes import build_nodes

TITLES = {"tracks": "SAM tracks", "graph": "graph nodes", "gt": "ground truth"}
METRICS = {"tracks": "tracks, no graph", "graph": "graph"}  # names in eval/metrics.json
ALPHA = 0.3  # mask fill opacity
OUTLINE = 2  # mask outline width in pixels
HEADER = 28  # pixels above each panel for its title

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--source", nargs="+", default=[f"replica/{scene}/00" for scene in replica.SCENES], help="as for run.py"
)
parser.add_argument("--stride", type=int, default=20)
parser.add_argument("--mask-source", default="sam3", choices=["sam3", "gt"])
parser.add_argument("--pose-source", choices=["gt", "vggt"], help="as for run.py")
parser.add_argument("--vocab", help="as for run.py")
parser.add_argument("--panels", nargs="+", default=["tracks", "graph"], choices=list(TITLES))
parser.add_argument("--width", type=int, default=800, help="width of one panel in pixels")
parser.add_argument("--fps", type=float, default=4)
parser.add_argument("--show-ignored", action="store_true", help="also draw what the evaluation ignores")
args = parser.parse_args()


def palette(n: int, seed: int) -> np.ndarray:
    """(n, 3) bright random colours, light enough for black text."""
    return np.random.default_rng(seed).uniform(90, 255, (n, 3)).astype(np.uint8)


def panel_sources(run: Run) -> dict[str, tuple[Masks, np.ndarray, dict, dict]]:
    """Per panel: the masks, an ID per mask (-1 to leave it out), and a label and colour per ID."""
    hidden = set() if args.show_ignored else replica.IGNORE_CLASSES
    masks = load_stage(run.stage_dir("masks"), Masks)
    vocab = json.loads((run.stage_dir("masks") / "metadata.json").read_text())["vocab"]
    track_colour = palette(int(masks.sam_id.max()) + 1, seed=0)
    out = {}
    if "tracks" in args.panels:
        labels = {int(t): vocab[p] for t, p in zip(masks.sam_id.tolist(), masks.prompt.tolist(), strict=True)}
        out["tracks"] = masks, masks.sam_id, labels, {t: track_colour[t] for t in labels}
    if "graph" in args.panels:
        graph = load_stage(run.stage_dir("graph"), Graph)
        boxes = load_stage(run.stage_dir("boxes"), Boxes)
        nodes = build_nodes(masks, boxes, graph, vocab)
        colours = {n.id: track_colour[np.bincount(masks.sam_id[n.rows]).argmax()] for n in nodes}
        out["graph"] = masks, graph.node, {n.id: n.label for n in nodes}, colours
    if "gt" in args.panels:
        gt = load_stage(replace(run, mask_source="gt", vocab="").stage_dir("masks"), Masks)
        objects = replica.load_objects(replica.parse_source(run.source)[0])
        ids = np.where(rle_area(gt.rle) >= ground_truth.MIN_PIXELS, gt.sam_id, -1)
        colour = palette(max(objects) + 1, seed=1)
        out["gt"] = gt, ids, {o: objects[o].class_name for o in objects}, {o: colour[o] for o in objects}
    for name, (m, ids, labels, colours) in out.items():
        shown = np.array([i >= 0 and labels[i] not in hidden for i in ids.tolist()], bool)
        out[name] = m, np.where(shown, ids, -1), labels, colours
    return {name: out[name] for name in args.panels}


def draw(rgb: np.ndarray, masks: Masks, ids: np.ndarray, labels: dict, colours: dict, frame: int, title: str, font):
    """One panel: rgb with a filled and outlined mask and a "label #ID" per ID in the frame, under a title strip.

    A label sits on the mask pixel nearest the mask's centroid, moved inside the panel when it would stick out.
    """
    rows = np.flatnonzero((masks.frame == frame) & (ids >= 0))
    objs = []  # (union of the ID's masks, ID)
    for i in np.unique(ids[rows]).tolist():
        rles = [masks.rle[r] for r in rows[ids[rows] == i]]
        objs.append((mask_utils.merge(rles) if len(rles) > 1 else rles[0], i))
    objs.sort(key=lambda o: -int(mask_utils.area(o[0])))  # large ones first, so small ones stay visible on top

    size = (args.width, round(rgb.shape[0] * args.width / rgb.shape[1]))
    image = np.asarray(Image.fromarray(rgb).resize(size), dtype=np.float32)
    anchors = []  # (x, y, ID) of each label
    for rle, i in objs:
        m = np.asarray(Image.fromarray(mask_utils.decode(rle) * 255).resize(size, Image.Resampling.NEAREST)) > 0
        if not m.any():
            continue
        image[m] = (1 - ALPHA) * image[m] + ALPHA * colours[i]
        image[m & ~binary_erosion(m, iterations=OUTLINE, border_value=1)] = colours[i]  # none on the frame edge
        ys, xs = np.nonzero(m)
        k = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)
        anchors.append((int(xs[k]), int(ys[k]) + HEADER, i))

    panel = Image.new("RGB", (size[0], size[1] + HEADER))
    panel.paste(Image.fromarray(image.astype(np.uint8)), (0, HEADER))
    d = ImageDraw.Draw(panel)
    d.text((8, HEADER // 2), title, font=font, fill="white", anchor="lm")
    for x, y, i in anchors:  # after all masks, so no mask paints over a label
        text = f"{labels[i]} #{i}"
        left, top, right, bottom = d.textbbox((x, y), text, font=font, anchor="mm")
        x += max(0, 2 - left) - max(0, right + 2 - panel.width)
        y += max(0, HEADER + 1 - top) - max(0, bottom + 1 - panel.height)
        left, top, right, bottom = d.textbbox((x, y), text, font=font, anchor="mm")
        d.rectangle([left - 2, top - 1, right + 2, bottom + 1], fill=tuple(colours[i].tolist()))
        d.text((x, y), text, font=font, fill="black", anchor="mm")
    return np.asarray(panel)


font = ImageFont.load_default(size=13)
for source in args.source:
    try:
        run = footage.make_run(source, args.stride, args.mask_source, args.pose_source, vocab.tag(args.vocab))
    except ValueError as e:
        parser.error(str(e))
    if not footage.is_replica(run) and "gt" in args.panels:
        raise SystemExit(f"{source} has no ground truth to show: leave gt out of --panels")
    sources = panel_sources(run)
    metrics_path = run.stage_dir("eval") / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    titles = {}
    for name, (_, ids, _, _) in sources.items():
        titles[name] = f"{TITLES[name]}: {len(np.unique(ids[ids >= 0]))} IDs"
        if METRICS.get(name) in metrics:
            titles[name] += f", IDF1 {metrics[METRICS[name]]['idf1']:.2f}"

    video = []
    for frame in footage.frame_numbers(run).tolist():
        rgb = footage.load_rgb(run, frame)
        panels = [draw(rgb, *src, frame, titles[name], font) for name, src in sources.items()]
        canvas = np.concatenate(panels, axis=1)
        ImageDraw.Draw(img := Image.fromarray(canvas)).text(
            (canvas.shape[1] - 8, HEADER // 2), footage.frame_label(run, frame), font=font, fill="white", anchor="rm"
        )
        canvas = np.asarray(img)
        video.append(canvas[: canvas.shape[0] // 2 * 2, : canvas.shape[1] // 2 * 2])  # H.264 wants even sizes

    path = run.stage_dir("eval") / f"{'_'.join(args.panels)}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = torch.from_numpy(np.stack(video)).permute(0, 3, 1, 2).contiguous()
    VideoEncoder(frames, frame_rate=args.fps).to_file(path, codec="libx264", pixel_format="yuv420p", crf=23)
    print(f"Saved {len(video)} frames to {path}")
