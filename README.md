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

## Results

All 8 Replica scenes at stride 10. SAM 3 is prompted with the `indoor` list, not the scenes' class names. The GT rows
use Replica's object masks instead of SAM 3's, as an upper bound. The maps compared:

- **SAM's tracks**: each SAM 3 track is an object, with no memory or refine (`evaluate_openlex3d.py --variants
  tracks`; `run.py` always scores them).
- **memory**: the object memory before refine (`evaluate_openlex3d.py --variants memory`).
- **graph**: the full graph.
- **GT tracks**: each GT object, a perfect map.

Caveats: the parameters were tuned on these same scenes. The VGGT-Omega maps are moved into Replica's frame by the
GT camera path (ATE 2.4 cm), and that is used for scoring only.

**OpenLex3D queries**, all queries, AP in percent. Objects is the total over the 8 scenes:

| Masks | Poses | Map | Objects | mAP | AP50 | AP25 |
|---|---|---|---|---|---|---|
| ConceptGraphs (paper) | | | | 5.86 | 11.32 | 22.39 |
| HOV-SG (paper) | | | | 5.76 | 11.67 | 25.30 |
| OpenMask3D (paper) | | | | 11.47 | 17.01 | 24.02 |
| Kassab2024 (paper) | | | | 1.38 | 2.87 | 7.54 |
| SAM 3 | VGGT-Omega | SAM's tracks | 951 | 8.34 | 15.77 | 31.24 |
| SAM 3 | VGGT-Omega | memory | 980 | 9.45 | 18.38 | 35.74 |
| SAM 3 | VGGT-Omega | graph | 431 | 9.16 | 17.04 | 34.66 |
| SAM 3 | Replica | graph | 453 | 10.35 | 20.19 | 38.04 |
| GT | VGGT-Omega | GT tracks | 488 | 16.49 | 32.32 | 43.99 |
| GT | VGGT-Omega | graph | 424 | 15.83 | 29.06 | 40.12 |
| GT | Replica | GT tracks | 488 | 19.05 | 37.26 | 48.28 |
| GT | Replica | graph | 464 | 18.72 | 34.40 | 45.77 |

- **SAM's tracks already beat ConceptGraphs** (8.34 against 5.86 mAP). The memory adds 1.1 and refine takes 0.3 back.
  So most of the gain comes from the newer models (SAM 3 and SigLIP 2, where ConceptGraphs uses SAM and CLIP), not
  from the graph.
- **Masks are the largest gap.** GT masks raise the graph from 9.16 to 15.83 with the same poses. Replica's poses add
  1.2 to 2.9.
- **With perfect masks the graph costs little:** 0.3 to 0.7 below the GT tracks.

**OpenLex3D segmentation, top 5**, as shares of the GT objects' points (SAM 3 with VGGT-Omega poses, unless the row
says GT):

| Map | Synonym | Wrong | Clutter | Missing |
|---|---|---|---|---|
| ConceptGraphs | 0.408 | 0.214 | 0.237 | 0.017 |
| SAM's tracks | 0.558 | 0.134 | 0.105 | 0.134 |
| memory | 0.585 | 0.118 | 0.114 | 0.119 |
| graph | 0.510 | 0.072 | 0.099 | 0.264 |
| GT tracks, Replica poses | 0.644 | 0.140 | 0.111 | 0.020 |
| graph on GT masks, Replica poses | 0.575 | 0.135 | 0.109 | 0.100 |

Refine trades coverage for precision: it halves the objects and cuts wrong labels from 0.118 to 0.072, but raises
missing from 0.119 to 0.264. Walls, floors and ceilings are not scored, and missing is averaged per object, so every
small object refine drops counts in full. Even with GT masks it drops 6 to 9 real objects per scene: ceiling lights,
wall plugs, vents and signs.

**`run.py`**, mean over the scenes (IDs is per scene):

| Masks | Poses | Map | IDF1 | IDs | kept% | L0 mAP | L1 mAP | reach | nearest, oracle |
|---|---|---|---|---|---|---|---|---|---|
| SAM 3 | VGGT-Omega | SAM's tracks | 0.47 | 119 | 0.49 | 0.54 | 0.70 | 0.82 | 0.41 |
| SAM 3 | VGGT-Omega | graph | 0.66 | 54 | 0.90 | 0.53 | 0.76 | 0.69 | 0.50 |
| SAM 3 | Replica | graph | 0.67 | 57 | 0.93 | 0.55 | 0.70 | 0.72 | 0.57 |
| GT | VGGT-Omega | GT tracks | 1.00 | 62 | 1.00 | 0.70 | 0.78 | 1.00 | 0.89 |
| GT | VGGT-Omega | graph | 0.96 | 53 | 0.99 | 0.63 | 0.77 | 0.86 | 0.69 |
| GT | Replica | GT tracks | 1.00 | 62 | 1.00 | 0.70 | 0.78 | 1.00 | 0.95 |
| GT | Replica | graph | 0.98 | 58 | 0.99 | 0.65 | 0.78 | 0.92 | 0.83 |

What the graph adds is identity over time, which OpenLex3D doesn't score: IDF1 goes from 0.47 to 0.66, and the share
of objects that keep their node when they come back into view goes from 0.49 to 0.90.
