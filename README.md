# eai-project

DD2600 course project: a 3D map of the objects in a video that can be searched by text. Each object is a node with a
3D box, a label and an embedding, and edges say which objects rest on or stand near which. The video is a Replica
scene, scored against its ground truth, or a phone video.

SAM 3 finds and tracks object masks, DINOv3 and SigLIP 2 embed each mask, and depth and camera poses (Replica's, or
VGGT-Omega's from the video alone) lift the masks to 3D. An object memory groups the masks into objects one frame at a
time, and once the whole video is in, only the objects whose box can be trusted are kept.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and FFmpeg.

```bash
uv sync --extra cu130   # RTX 50-series (Blackwell); cu126 for an H100 or a driver older than 580
uv run --no-sync python -m ipykernel install --user --name eai-project --display-name "EAI project"
uv run --no-sync hf auth login
```

Use `uv run --no-sync` for everything: a plain `uv run` re-syncs without the extra and replaces the CUDA build of
torch. Three models are gated: request access to [facebook/sam3](https://huggingface.co/facebook/sam3),
[facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) and
[facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) before logging in.

**Data.** `bash scripts/download_data.sh` downloads the 8 Replica scenes that OpenLex3D labels (about 46 GB) into
`data/vmap/` (frames, depth, poses and object masks, from vMAP) and `data/gt_openlex3d/` (OpenLex3D's labels and text
queries). Phone videos go in `data/phone/`.

## Code

```
run.py                   the pipeline: computes and caches every stage of a run, then scores it on Replica
evaluate_openlex3d.py    OpenLex3D's own scoring of our maps, next to the methods in its paper
notebooks/phone.ipynb    phone clips to a map: runs the pipeline, shows the map, the graph, the video and a search
notebooks/replica.ipynb  one Replica scene: scores, our boxes against the GT boxes, the video, a search
src/
  data/                  what a run is and where its data comes from
    stages.py              Run, the type of every cached stage, saving and loading them, mask helpers
    paths.py               where data/ and outputs/ are
    footage.py             the frames of any run: Replica's, or a phone video's (the frames stage)
    replica.py             Replica's depth, poses, GT masks, classes and OpenLex3D queries
    vocab.py               SAM's prompt lists (--vocab)
    cli.py                 the arguments that pick a run (--source, --stride, ...)
  models/                the per-mask stages
    sam.py                 masks and tracks (SAM 3)
    dino.py                appearance embedding per mask (DINOv3)
    siglip.py              text-comparable embedding per mask (SigLIP 2)
    vggt.py                depth and cameras from the video alone (VGGT-Omega)
    crops.py               the crops of a mask the encoders see
  geometry/
    depth.py               depth and camera of a frame, lifting pixels to 3D
    mask_boxes.py          a 3D box per mask, and which masks' depth to trust (the boxes stage)
  graph/                 the object graph
    memory.py              the object memory: masks to objects, one frame at a time
    refine.py              the whole video at once: split, label, merge, and keep the objects whose box holds up
    multiview.py           each object's box from every view of it
    build.py               the graph stage: memory, refine, boxes, saved
    nodes.py               the graph's nodes (label, box, masks) and their colours
    edges.py               "on" and "near" between nodes
    query.py               text search over the nodes
  evaluation/
    metrics.py             IDF1, revisits, text queries and the nearest-object query, against Replica's ground truth
    ground_truth.py        the ground truth they need, and the transform from a VGGT map to Replica's frame
  views/
    rerun_map.py           the map in Rerun: point cloud, camera path, boxes, edges (and the GT boxes)
    top_down.py            the graph from above, as a picture
    video.py               the video with every object outlined and labelled
```

## Running it

```bash
uv run --no-sync python run.py --source replica/room_0/00                   # one Replica scene
uv run --no-sync python run.py --vocab indoor --pose-source vggt            # all 8, fixed prompts, VGGT-Omega poses
uv run --no-sync python run.py --source phone/kitchen                       # data/phone/kitchen.mp4
uv run --no-sync python run.py --source replica/room_0/00 --graph-only      # rebuild only the graph
```

Each stage is cached under `outputs/<source>/stride<k>/`, so a rerun only computes what is missing or out of date.
The options:

- `--vocab`: SAM's prompts, `indoor` (102 common indoor objects) or a file with one prompt per line. Without it,
  Replica runs use each scene's own GT classes and phone runs `indoor`.
- `--pose-source`: `gt` (Replica's depth and poses) or `vggt` (from the video alone; always for phone video).
- `--stride`: use every k-th frame (default 20). VGGT-Omega fits 300 frames in one pass on a 32 GB GPU.
- `--mask-source gt`: Replica's GT masks instead of SAM's, as an upper bound.
- `--overwrite`: recompute stages even if they are cached.

### Memory

Measured on an RTX 5090 (32 GB) with 60 GB of RAM:

| Stage | GPU | RAM |
|---|---|---|
| SAM 3 | about 10 GB: its tracking state on the GPU, and 16 prompts per detector pass | grows with frames and objects: 4.7 GB over 150 frames with 89 objects |
| DINOv3, SigLIP 2 | fit a 10 GB GPU (masks are encoded 32 at a time) | small |
| VGGT-Omega | 4.6 GB + 76 MB per frame: 100 frames 12 GB, 200 frames 20 GB, 300 frames 27 GB | small |
| Graph (memory, refine, boxes) | none | under 2 GB, also for 249 frames and 4,371 masks |
| `evaluate_openlex3d.py` | none | about 11 GB per scene |

So VGGT-Omega sets the GPU you need: `MAX_FRAMES` in `models/vggt.py` (300) suits 32 GB; by the rule above a 10 GB GPU
takes about 60 frames, so lower it there and use a larger `--stride`. On a 10 GB GPU, also set `STATE_DEVICE` to `"cpu"`
and `DETECT_BATCH` to 1 in `models/sam.py`: SAM 3 then needs 4.5 GB, at half the speed. For RAM, the whole pipeline on a
249-frame phone clip ran within a 20 GB limit, and OpenLex3D's scoring takes about 11 GB per scene. Run one heavy job at
a time: five scoring runs at once ran a 60 GB machine out of memory. The cache takes about 100 MB per Replica scene at
stride 20, half of it VGGT-Omega's depth.

After a run, these read its cached stages (with the same arguments):

```bash
uv run --no-sync python -m graph.query --source replica/room_0/00 "striped pillow"      # search by text
uv run --no-sync python -m graph.query --source replica/room_0/00 cushion --near sofa   # nearest of a type
uv run --no-sync python -m graph.edges --source replica/room_0/00                       # list the edges
uv run --no-sync python -m views.top_down --source replica/room_0/00 --gt               # eval/edges.png
uv run --no-sync python -m views.rerun_map --source replica/room_0/00 --gt              # eval/map.rrd
uv run --no-sync python -m views.video --source replica/room_0/00 --panels tracks graph gt
```

## Evaluating on Replica

For maps made without the scenes' class names, at the frame rate of the methods in OpenLex3D's paper:

```bash
uv run --no-sync python run.py --vocab indoor --stride 10 --pose-source vggt
uv run --no-sync python evaluate_openlex3d.py --vocab indoor --stride 10 --pose-source vggt
uv run --no-sync python evaluate_openlex3d.py --table   # every method scored so far
```

**`run.py`** scores our graph and, as a baseline, SAM's raw tracks, per scene and as the mean over the scenes. It also
saves the scores to each run's `eval/metrics.json`:

- **IDF1**: is each real object one node over the whole video? **IDP** is its precision, **IDR** its recall.
  **IDs/obj** above 1 means duplicate nodes, and **obj/ID** above 1 means nodes that mix objects.
- **kept%**: when an object leaves the view and comes back, the share that keep their node.
- **L0 / L1 mAP**: text search with OpenLex3D's queries, naming the object ("throw pillow") or describing it
  ("striped pillow"). **reach** is the share of the queried objects that are in the map at all.
- **Nearest object, oracle**: does the nearest object of a type, by our boxes, match the ground truth's?

These score whole objects, so they aren't comparable with published numbers.

**`evaluate_openlex3d.py`** scores the same maps with [OpenLex3D](https://github.com/openlex3d/openlex3d)'s own
code, and prints them next to its paper's numbers for ConceptGraphs, HOV-SG, OpenMask3D and Kassab2024:

- **Queries** (the main comparison): each query's 10 best objects, scored with instance AP against the mesh
  (mAP, AP50, AP25).
- **Segmentation, top-5**: per GT point, whether the nearest predicted object's 5 best labels hold a synonym, are
  wrong, or whether the point is missing (no predicted point within 5 cm).

A VGGT-Omega map is first moved into Replica's frame, by the transform that best fits its camera path to Replica's.
That uses the GT cameras for scoring only; say so next to such numbers.

Setup, once: `uv sync --extra cu130 --extra openlex3d`. Scoring takes about 11 GB of RAM per scene, so run nothing
else heavy alongside. To check the scoring against the paper, download the methods' predictions and score one:

```bash
wget -P data/openlex3d_predictions http://aisdatasets.cs.uni-freiburg.de/openlex3d/predictions_sparse.zip   # 2 GB
unzip data/openlex3d_predictions/predictions_sparse.zip -d data/openlex3d_predictions/extracted
uv run --no-sync python evaluate_openlex3d.py --baselines concept-graphs   # gives the paper's 5.86 / 11.32 / 22.39
```
