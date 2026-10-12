# eai-project

DD2600 course project.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and FFmpeg (used by torchcodec for video decoding).

```bash
git clone https://github.com/johanhroos/eai-project.git
cd eai-project
uv sync --extra cu126   # or for blackwell: uv sync --extra cu130
uv run --no-sync python -m ipykernel install --user --name eai-project --display-name "EAI project"
```

Pick the extra that matches the GPU:

| Extra   | GPU                                  | Notes                                             |
|---------|--------------------------------------|---------------------------------------------------|
| `cu130` | RTX 50-series (e.g. RTX 5090)        | Blackwell needs this; cu126 has no sm_120 kernels |
| `cu126` | H100                                 | Also use it if `nvidia-smi` shows a driver < 580  |

## Check the environment

```bash
uv run --no-sync python3 -m smoke
```

It prints the torch/CUDA versions and GPU, then runs SDPA, `torch.compile`, and a video decode.

IMPORTANT: Use `uv run --no-sync` for everything: a plain `uv run` re-syncs without the extra and replaces the CUDA build of
torch with the default one from PyPI.

## Data

Approx 46GB file.
```bash
bash scripts/download_data.sh
```

Downloads the 8 Replica scenes that have OpenLex3D labels (room0–2, office0–4) into `data/`. Rerunning resumes an interrupted
download.

| Path | Source | Contents |
|---|---|---|
| `data/vmap/<scene>/imap/{00,01}/` | [vMAP](https://github.com/kxhit/vMAP) | Two camera paths × 2000 frames: `rgb/`, `depth/` (uint16, mm), `semantic_instance/` (Replica object ID per pixel), `traj_w_c.txt` (camera-to-world 4×4 per line). Path `00` is the NICE-SLAM trajectory. |
| `data/vmap/<scene>/habitat/info_semantic.json` | vMAP | Object ID → Replica class name |
| `data/gt_openlex3d/replica/<scene>/` | [OpenLex3D](https://github.com/openlex3d/openlex3d) | Open-vocabulary labels and text queries per object ID |

Camera: 1200×680, fx = fy = 600, cx = 599.5, cy = 339.5. Frames are numbered `0`–`1999` without zero padding, so sort
by number. Scene names are `room_0` in vMAP and `room0` in OpenLex3D.

## SAM 3 preview

Request access to [facebook/sam3](https://huggingface.co/facebook/sam3) and authenticate on the GPU machine
with HuggingFace in terminal once.

```bash
uv run --no-sync hf auth login
```

Try running the cells in `notebooks/explore_data.ipynb`
with the `EAI project` kernel.

## Pipeline

`run.py` computes these stages for each run and caches each one under `outputs/<source>/stride<k>/`, so a rerun only
computes what is missing, or what was computed from a stage that has changed since. Every per-mask stage has one row
per row of `masks`.

The `features` stage needs access to [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
as well: request it like SAM 3 and use the same `hf auth login`.

| Stage | Code | Contents |
|---|---|---|
| `frames` | `utils/footage.py` | Only for phone videos: every stride-th frame as a JPEG |
| `masks` | `models/sam.py`, `utils/replica.py` | SAM 3 masks prompted with the scene's classes or a fixed list (`--mask-source sam3`), or the GT masks (`gt`) |
| `features` | `models/dino.py` | DINOv3 embedding per mask: patch tokens of the mask's padded box crop, averaged over the mask |
| `siglip` | `models/siglip.py` | SigLIP 2 embedding per mask, comparable with text: the mean of a square crop around the mask and the same crop greyed outside it |
| `vggt` | `models/vggt.py` | Only with `--pose-source vggt`: VGGT-Omega depth and cameras for every frame, in metres with z up |
| `boxes` | `utils/lift3d.py` | World-frame 3D box per mask and 256 of its points, from depth and pose (Replica's, or VGGT-Omega's). An object's box spans its masks' pooled points (`graph/node.py`), leaving out masks far from the rest |
| `graph` | `graph/association.py`, `graph/pipeline.py` | Object node per mask, from an object memory that takes the masks one frame at a time and matches them by DINO appearance and location. Also the memory itself, the match scores and the node count over time. `graph/node.py` gives each node's label and box |
| `events`, `gt_boxes` | `utils/lift3d.py` | Ground truth for the evaluation: when objects leave the view and come back, and one 3D box per object |

The graph stage (`graph/pipeline.py`, `graph/association.py`) builds an object memory one frame at a time:

1. Masks of one frame that overlap strongly (mask IoU >= 0.6) are one observation: one object found by several prompts.
2. Each observation is scored against every object in memory with its **DINO appearance** and its **location**:
   the best DINO cosine against the object's few stored descriptors, and the overlap and centre distance of the 3D
   boxes. Objects more than 1 m away aren't candidates. It joins the best object if the score is above the threshold,
   and otherwise becomes a new object. SAM IDs and labels don't decide it: two SAM IDs on one object end up in one
   node when DINO and location agree, and an object that leaves the view and comes back keeps its node.
3. The object's **box grows to the union** of every box it was matched with (the smallest minimum and largest maximum
   per axis), so it covers the whole space the object fills, also the parts hidden in some views. It never shrinks.
   It keeps up to four DINO descriptors from different views, and sums its observations' SigLIP embeddings for text
   search. SigLIP takes no part in matching.

The weights are DINO 0.45 and location 0.40 (normalised to 0.53 and 0.47), and the threshold 0.55 is above each, so a
match needs both: the same place and the same look (`configs/association.json`, `--association-config` for another).

```bash
uv run --no-sync python run.py                                                # SAM 3 masks, every scene
uv run --no-sync python run.py --source replica/room_0/00 --mask-source gt    # GT masks
uv run --no-sync python run.py --source replica/room_0/00 --overwrite graph   # recompute only the graph
uv run --no-sync python run.py --pose-source vggt                             # VGGT-Omega depth and poses
uv run --no-sync python -m graph.query --source replica/room_0/00 "striped pillow"   # search a run's graph
uv run --no-sync python -m graph.query --source replica/room_0/00 cushion --near sofa   # nearest object of a type
uv run --no-sync python -m graph.query --source replica/room_0/00 --position 3.6 2.5 -0.9   # objects nearest a point
uv run --no-sync python -m graph.edges --source replica/room_0/00                      # list the graph's edges
uv run --no-sync python -m graph.plot_edges --source replica/room_0/00                 # eval/edges.png, nodes and edges
```

`notebooks/graph.ipynb` shows one run's nodes and edges from above (`graph/plot_edges.py`), one node with its
neighbours, a nearest-object query, the map in Rerun with the edges as lines (`world/edges/on` and `world/edges/near`,
also in `utils/map_view.py`'s other uses), and the query scores.

`graph/edges.py` connects the nodes by their 3D boxes, computed on the fly like the nodes:

- **on**: one object rests on or in another, such as a cushion on a sofa or a vase on a cabinet. Most of its footprint
  lies over the other's, its bottom is above the other's bottom and near its top, and it is the smaller one.
- **near**: the boxes are at most 25 cm apart.

Walls, floors, ceilings and nodes with a box side over 3 m get no edges, since they touch everything. On room_0 the
84 nodes get 31 "on" and 115 "near" edges. `graph.query` lists each result's "on" and "under" edges, which tells
similar objects apart, and `--near` ranks the objects of a type by distance to the best match of another text: the
smallest gap between the boxes, then the closest centres. The type is the SAM label when the text is one of the
prompts, since SigLIP's scores for a type are close together, and otherwise SigLIP's `--top` best matches.

`--overwrite` with no stage names recomputes every stage, SAM included.

### VGGT-Omega poses

With `--pose-source vggt`, the 3D boxes and the graph use depth and cameras from
[VGGT-Omega](https://github.com/facebookresearch/vggt-omega) instead of Replica's, and nothing from the ground truth.
VGGT-Omega reconstructs all frames of a run in one forward pass, in its own units and in the first camera's frame.
`models/vggt.py` then puts it in metres with z up:

- **Up**: perpendicular to every camera's x axis. That holds when the camera isn't rolled, as with a handheld phone.
  On the 8 Replica scenes it is 0.2 to 0.7 degrees off.
- **Scale**: the median ratio of [Depth Pro](https://huggingface.co/apple/DepthPro-hf)'s metric depth to VGGT's
  over 8 frames. On Replica it is 5.6% off on average and 17.5% at worst (office_1). Depth Anything V2 metric indoor
  was 33% off.
- **Heading**: along the walls. The normals of vertical surfaces in VGGT's depth point in four directions 90 degrees
  apart in a room, and the map is turned so they lie on x and y. On Replica it is 0 to 1.2 degrees off the walls.
  Before, the first camera's view was +x, and a painting along a wall at an angle got a 66 cm thick box.

On Replica the `vggt` stage's `metadata.json` also scores this against the GT trajectory, for scoring only: the
camera positions are 1.6 to 2.6 cm off (ATE after a best-fit similarity transform) and depth 0.8 to 1.1% (AbsRel).
IDF1 and query mAP come out the same as with Replica's poses (0.58, 0.44 and 0.52 over the 8 scenes).

- The checkpoints are gated like SAM 3: request access to [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega).
  VGGT-Omega is under the FAIR Noncommercial Research License and Depth Pro under Apple's ML Research license.
- `uv sync` installs the model code from GitHub, pinned to a commit. It asks for numpy<2, which `pyproject.toml`
  overrides.
- 100 frames take about 10 s and 13 GB of GPU memory on an RTX 5090, and 286 frames 51 s and 28 GB. Above 250
  frames, `run.py` stops before SAM and asks for a larger `--stride`.

### Phone video

Put a video in `data/phone/`, such as `data/phone/kitchen.mp4` (any format ffmpeg reads; portrait clips are turned
upright), and run it as `phone/kitchen`. There is no ground truth, so it uses VGGT-Omega's poses and stops after the
graph:

```bash
uv run --no-sync python run.py --source phone/kitchen
uv run --no-sync python -m graph.query --source phone/kitchen "coffee mug"
uv run --no-sync python -m utils.video --source phone/kitchen
uv run --no-sync python -m utils.map_view --source phone/kitchen   # eval/map.rrd for the Rerun viewer
uv run --no-sync python -m graph.plot_edges --source phone/kitchen # eval/edges.png, the objects and their edges
```

`notebooks/map.ipynb` does all of this for one set of clips at a time: it collects clips sent with Tailscale, runs
the pipeline with the prompts you set, and shows the map in Rerun (the point cloud, the camera's path and the objects'
3D boxes), a picture of the graph from above with its edges (`graph/plot_edges.py`), the labelled video and a text
search. 

- **Several videos of one place** make one map: `--source phone/IMG_0644+IMG_0645`. SAM tracks each video on its
  own, so no ID carries across the cut, VGGT-Omega reconstructs all their frames in one pass, so they share one 3D
  frame and scale, and the graph's pass 2 links the objects they share. On two clips of one room (20 s together), the
  shoes from both clips landed within 5 cm of each other, and 5 of the 18 objects were linked across the clips: both
  shoes, the ladder (one clip only saw its feet) and two shirts, all right by eye. Together the videos must stay under
  250 frames after the stride.
- The `frames` stage keeps every `--stride`-th frame (stride 20 at 30 fps is 1.5 frames per second) and scales it
  down to 1280 pixels on the long side. A 60 s clip gives 90 frames.
- SAM is prompted with a fixed list of 60 indoor objects (`utils/vocab.py`). `--vocab` takes another list, by name
  (`indoor`) or as a file with one prompt per line, and puts that run's outputs in their own directories. Replica runs
  keep the scene's GT classes unless `--vocab` is given, so `--vocab indoor` on Replica is the test without that leak.
- HDR video (most iPhone video by default) is decoded without tone mapping, so colours look flat; the `frames` stage
  warns about it. Record in SDR if you can.
- Tested on a video made from room_0's frames: the whole run took 7 minutes, mostly SAM. Scored against room_0's
  ground truth, the graph gets IDF1 0.60 and L0/L1 mAP 0.49/0.60, against 0.62 and 0.50/0.62 for the Replica run
  with GT classes and poses (one scene, the one the graph was tuned on). The same frames as a 1080p 10-bit HEVC HLG
  video, as an iPhone records, come out at 1280 x 720 with the camera 1.5 cm off (ATE), scale 1.4% off and up 0.2
  degrees off.
- It has not been tried on real phone footage yet: motion blur, exposure changes and rolling shutter are untested,
  depth confidence isn't used yet, and SAM has only run on 100 frames.

### Evaluation

After the stages, `run.py` scores two sets of object IDs against the GT instances and prints two tables: the raw SAM
tracks and the graph. The scores are also saved to `outputs/<source>/stride<k>/<mask>_<pose>/eval/metrics.json`.

The first table scores identities. Walls, floors, ceilings and unlabelled objects are left out.

- **IDF1** (identity F1): pairs predicted IDs one-to-one with GT objects over the whole run. Both splitting an object
  over several IDs and merging different objects into one ID lower it. IDP and IDR are its precision and recall.
- **IDs/obj** and **obj/ID**: how many predicted IDs each GT object gets, and how many GT objects each ID covers.
- **Revisits**: for every time an object leaves the view and comes back (the `events` stage), is its ID the same
  afterwards (**kept**) or not (**switched**)? **missed** means no prediction matches the object on one side.

The second table scores the map as a text search (`graph/query.py`). Each object gets the mean of its masks' SigLIP
embeddings, and each [OpenLex3D](https://github.com/openlex3d/openlex3d) query ranks all objects by cosine similarity.
An object stands for the GT object its masks match in the most frames.

- **mAP**: mean average precision over the queries. **top1**: share of queries whose first answer is right.
  A duplicate (a second object for a GT object already found) counts as a wrong answer, but it only lowers mAP when it
  ranks above a target not found yet, and top1 ignores it. So these scores mostly see missing and wrongly merged
  objects; the first table's IDP and IDs/obj show duplicates. Synonyms make many queries for one object, so objects
  with many synonyms weigh more.
- **L0** queries name the object ("throw pillow"); **L1** add a pattern, material or picture on it ("stripe pillow").
- **reach**: share of the queried objects that some object in the map stands for, the recall a perfect ranking gets.

This scores whole objects; OpenLex3D's own protocol scores point clouds, so the numbers aren't comparable with the paper.

The third table scores the nearest-object query (`graph.query --near`) against the GT boxes (`evaluate.nearest_objects`).
Every GT object asks for the nearest object of every class with at least two other objects in the scene, and the answer
is the one nearest by the GT boxes. In the map, the anchor is the node that stands for the GT object, and the query is
right when the nearest candidate stands for the answer.

- **labels**: the candidates are the nodes with that SAM label, as `--near` does it.
- **oracle**: the candidates are the nodes that stand for an object of that class, which scores only the map's boxes
  and identities.
- **anchor**: share of the queries whose anchor is in the map. **reach**: whose anchor and answer both are, the most
  oracle can get.

With GT masks all four are close to 1, which checks the metric itself; not exactly 1, since the map's boxes come from
every 20th frame and the GT boxes from all frames, which flips a few near ties.

With `--mask-source gt` the tracks are the GT objects. Their identity row must score IDF1 = 1, which checks the
evaluation itself, and their query row shows what SigLIP gets from a perfect map.

### Video

`utils/video.py` renders a run's frames to an MP4 with each object's mask outlined and labelled with its class and ID, with
the SAM tracks and the graph nodes side by side. Run `run.py` first, since it reads the cached stages. A node takes the
colour of its largest track, so duplicates that the graph merged have two colours on the left and one on the right.

```bash
uv run --no-sync python -m utils.video --source replica/room_0/00                            # eval/tracks_graph.mp4
uv run --no-sync python -m utils.video --source replica/room_0/00 --panels tracks graph gt   # add the ground truth
```

### Graph: configuration, rebuilds and tuning

`configs/association.json` holds every matching parameter. Weights are normalised over the enabled cues
(`enable_dino`, `enable_geometry`), and a missing cue scores zero without redistributing its weight. Geometry is
`geometry_iou_weight * IoU + (1 - geometry_iou_weight) * exp(-centre distance / center_distance_scale)` with the
unpadded `lift3d.box_iou_3d`. A match needs a score strictly above the threshold; in each frame the strongest pairs
are assigned first, one observation per object. `require_same_label` adds a same-label rule, off by default since
prompt labels change between views.

Changing only the config recomputes only the graph. To rebuild it from the cached stages without loading any model
or scoring it:

```bash
uv run --no-sync python run.py --source replica/room_0/00 --graph-only --association-config configs/association.json
```

The graph stage directory holds the node per mask (`arrays.npz`, read by the evaluation, video and plots), `memory/`
(the compact object memory: IDs, labels, boxes, DINO descriptors, SigLIP sums, first and last frame;
`graph.pipeline.load_memory()` restores it and can go on with later frames), and `matches.json` and `history.json` (each
observation's best candidate and scores, and the node count per frame, for inspection). `--skip-text-eval` scores
identities without loading the text encoder.

`notebooks/visualize_graph.ipynb` (`graph/visualize.py`) is for tuning: interactive Plotly boxes and edges, RGB/SAM
overlays, frame and object selectors, the score of each match, and the node count over time. Changing a frame replays
the memory up to it. Adjust the weights, threshold, cues and geometry controls, click **Rebuild from cache** (CPU only,
no models), and **Save graph + config** to write the graph and a reusable config. On a headless cluster:

```bash
uv run --no-sync jupyter lab --no-browser --ip=127.0.0.1
```

Tests:

```bash
PYTHONPATH=src uv run --no-sync python -m unittest discover -s tests -v
```
