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
| `frames` | `data/footage.py` | Only for phone videos: every stride-th frame as a JPEG |
| `masks` | `models/sam.py`, `data/replica.py` | SAM 3 masks prompted with the scene's classes or a fixed list (`--mask-source sam3`), or the GT masks (`gt`) |
| `features` | `models/dino.py` | DINOv3 embedding per mask: patch tokens of the mask's padded box crop, averaged over the mask |
| `siglip` | `models/siglip.py` | SigLIP 2 embedding per mask, comparable with text: the mean of a square crop around the mask and the same crop greyed outside it |
| `vggt` | `models/vggt.py` | Only with `--pose-source vggt`: VGGT-Omega depth and cameras for every frame, in metres with z up |
| `boxes` | `geometry/mask_boxes.py` | World-frame 3D box per mask, 256 of its points and the median confidence of its depth (VGGT-Omega's; NaN for Replica's), from depth and pose. `mask_boxes.reliable` says which masks' depth to trust |
| `graph` | `graph/memory.py`, `graph/build.py` | Object node per mask, from an object memory that takes the masks one frame at a time: a mask joins its SAM track's node, and DINO appearance and location cut tracks that jump and match new ones. Then the nodes are split, labelled and merged as a whole, and only those whose box can be trusted are kept (`graph/refine.py`; the masks of dropped ones get node -1), and each node's box is drawn from every view of it (`graph/multiview.py`). Also the memory itself, the match scores, the merges, the node count over time and what refine did (`refine.json`). `graph/nodes.py` gives each node's label |
| `events`, `gt_boxes` | `geometry/mask_boxes.py` | Ground truth for the evaluation: when objects leave the view and come back, and one 3D box per object |

The graph stage (`graph/build.py`, `graph/memory.py`) builds an object memory one frame at a time. SAM 3's
tracker is mostly right about which masks are one object, so its tracks are the starting point, and DINO appearance
and 3D location check them:

1. Masks of one frame that overlap strongly (mask IoU >= 0.6) are one observation: one object found by several prompts.
   Its SAM IDs all join one node, the older one where they already had two. Those two nodes don't merge there: that put
   three monitors up to 2.7 m apart into one node on IMG_0649.
2. Only **reliable** masks are evidence (`mask_boxes.reliable`): enough depth points, and a median VGGT-Omega depth
   confidence of at least 2.5 (`mask_boxes.MIN_CONFIDENCE`). Far views fail it: their depth smears along the camera ray,
   so a 1 m umbrella seen from 5 m got a 2 m box. An unreliable mask joins its track's node and does nothing else.
3. A mask whose SAM ID has a node **joins that node**, unless the track was **cut**: its box is more than 0.3 m from
   the box of the track's last reliable mask, or its DINO cosine to that mask is under 0.5 and the two boxes don't
   touch, or another node matches it by 0.2 better than its own (a track that slid onto a neighbour). SAM 3 switches
   IDs between objects that look alike and between neighbours, and each cue finds switches the others miss: on Replica
   the gap and DINO alone find 69% of them, and wrongly cut 0.7% of the steps between masks of one object, which step 5
   can join again. A DINO drop alone, with the boxes still touching, is mostly the same object seen from another side:
   cutting there too found 73% of the switches but cut 3.2% of the steps within one object, and on IMG_0649 it made
   new nodes that flickered in and out of the video. In a frame where three or more tracks, and at least half, move
   over 0.3 m at once, the camera pose is off and nothing is cut. A node takes one mask per frame: when two masks of a
   frame follow their tracks to one node (one track was cut and its rest matched there), they are two objects, and the
   one that matches the node less is cut.
4. New tracks, and the rest of cut ones, are scored against every object in memory with their **DINO appearance**
   and **location**: the best DINO cosine against the object's few stored descriptors, and how much of one 3D box lies
   inside the other and how close their centres are. Objects more than 1 m away aren't candidates. One joins the best
   object above the threshold, and otherwise becomes a new object.
5. A node that gets a reliable view is scored the same way against the other nodes, and **merges** into the best one
   above the threshold, unless the two were ever seen in one frame. This joins an object that comes back under a new
   SAM ID, one first seen too far away to match, and a track that was cut by mistake.
6. While the memory is built, an object's box is around the pooled points of its reliable masks, without the outer
   2% on each side (`graph.nodes.node_box`); with no reliable mask, it is the box of its most confident one. It keeps
   up to four DINO descriptors from different views, and sums its observations' SigLIP embeddings for text search.
   SigLIP takes no part in matching.
7. Once every frame is in, the nodes are checked as a whole (`graph/refine.py`). The scene is static, so an object's
   place and extent are fixed; like ConceptGraphs' post-processing, this keeps fewer objects rather than more, and
   only those whose box can be trusted. A node whose points form clusters over 0.3 m apart is **split**, one node per
   cluster (10 cm cubes that hold at least a tenth of the node's typical cube; each mask goes with the cluster that
   holds most of its points): on IMG_0649 this separates two monitors a SAM track slid between. Each node is
   **labelled** from all its views: SAM's votes for its prompts, weighed by how well SigLIP's mean embedding of the
   node matches each prompt's text (73% of the Replica nodes get their GT object's class, against 70% from the votes
   alone). Nodes of one label then **merge** three ways: two masks of one object in the frames both are in, such as a
   chair and its seat (the smaller mask inside the other in the median shared frame); parts alike in DINO with half
   the points of the one seen less within 15 cm of the other's (ConceptGraphs' overlap; a fifth of its box inside the
   other's is enough, since a monitor's box can be 0.1 or 0.45 m deep with VGGT-Omega's depth); and parts that the
   frames seeing both agree are one, as in MaskClustering's view consensus (one mask holds both in 80% of at least 3
   frames). Two nodes seen in one frame as two masks don't merge, unless that is a slip of SAM's tracker: at most 20%
   of the frames of the one seen less (on IMG_0649, a monitor's two IDs shared 3 of 21 frames, where one track slid
   onto the next monitor). Checked alone against Replica's objects, 89% of the first, 41 of 42 of the second and 14 of
   14 of the third joined one object. Last, only nodes whose box can be trusted are **kept**: at least 10 reliable
   views, spanning at least 20 degrees around the object, and at least 70% **coverage**, the share of its points in
   view that lie inside its own mask in its frames. A node that joined several objects fails this (a far monitor and a
   table top SAM called a monitor, whose box spanned the room on IMG_0649), and a node seen from one direction has a
   depth extent no view checked. Nodes only ever small (under 0.2% of the image) and far (over 4 m) go too. The masks
   of dropped nodes get node -1, which the video, the map, the queries and the evaluation leave out. On the 8 Replica
   scenes with VGGT-Omega's depth, a scene keeps 46 nodes instead of the memory's 121, 7 instead of 48 of them
   matching no GT object; 5% of the object boxes kept are off (IoU under 0.2) instead of 21%, mean IoU 0.54 instead of
   0.45; of the 368 revisits (an object leaves the view and comes back), the merges keep the object's node in 224 and
   switch it in 34, against 214 and 47 for the memory alone, and after the last step 203 keep it and 17 switch while
   38 more fall on dropped nodes; IDF1 0.598 to 0.640 (precision 0.55 to 0.68, recall 0.67 to 0.61) and L1 mAP 0.552
   to 0.596. The price is the objects whose box couldn't be checked: 18.5 objects a scene get a good box (IoU 0.5 or
   more) instead of 25, and the nearest-object query, which needs both objects in the map, drops (oracle 0.51 to
   0.33). On IMG_0649, 43 nodes instead of 139, with 22% of the masks left out: most of the dropped nodes are chairs
   and monitors seen from under 20 degrees. The long tables left (6 to 8 m) follow rows of desks; whether each is one
   row, or two that one SAM track joined, no view can tell, since the rows are never in view together. The thresholds
   are refine.py's constants.
8. Then each object's **box** is drawn again from every view of it (`graph/multiview.py`), and the graph stage keeps it.
   Each reliable mask is lifted again from its frame's depth at the depth's own resolution (every pixel the mask
   covers more than half of, minus each view's outliers), not from the boxes stage's 256-point samples of the
   upsampled depth, so each view weighs by its pixels on the object and close views count more. A point lifted from
   one view stays only if it lands inside the object's mask in at least half of its other views where it is in the
   image and not hidden. A view whose depth smeared the object along its camera ray (a far view, or a thin object
   against the wall behind) puts points beside the object in the views from other directions. The box is around the
   kept points without the outer 0.5%: the old 2% cut the ends off long objects, which few views see. On the 8 Replica
   scenes with VGGT-Omega's depth, the boxes of the memory's object nodes (before step 7) overlap the GT boxes more
   (IoU 0.38 to 0.45, both grown by 5 cm on each side; 0.41 to 0.51 with Replica's depth; voting with the samples gave
   0.43 and 0.48), and the nearest-object query gets 0.31 and 0.51 (labels, oracle) instead of 0.29 and 0.49. Lifting
   like the map's point cloud (depth edges dropped, one point per 2 cm, no per-view outlier filter) gave 0.41: the map
   has the same streaks, they just look like a few specks among the surfaces. On IMG_0649, 6 monitors seen in 20 or
   more masks had boxes 2 to 3 m long, and 2 were still over 1.5 m: a node that holds two monitors (1.8 m), which step
   7 now splits, and something 3.2 m tall that SAM called a monitor.

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
uv run --no-sync python -m views.top_down --source replica/room_0/00                 # eval/edges.png, nodes and edges
```

`notebooks/graph.ipynb` shows one run's nodes and edges from above (`views/top_down.py`), one node with its
neighbours, a nearest-object query, the map in Rerun with the edges as lines (`world/edges/on` and `world/edges/near`,
also in `views/rerun_map.py`'s other uses), and the query scores.

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
uv run --no-sync python -m views.video --source phone/kitchen
uv run --no-sync python -m views.rerun_map --source phone/kitchen   # eval/map.rrd for the Rerun viewer
uv run --no-sync python -m views.top_down --source phone/kitchen # eval/edges.png, the objects and their edges
```

`notebooks/map.ipynb` does all of this for one set of clips at a time: it collects clips sent with Tailscale, runs
the pipeline with the prompts you set, and shows the map in Rerun (the point cloud, the camera's path and the objects'
3D boxes), a picture of the graph from above with its edges (`views/top_down.py`), the labelled video and a text
search. 

- **Several videos of one place** make one map: `--source phone/IMG_0644+IMG_0645`. SAM tracks each video on its
  own, so no ID carries across the cut, VGGT-Omega reconstructs all their frames in one pass, so they share one 3D
  frame and scale, and the graph's pass 2 links the objects they share. On two clips of one room (20 s together), the
  shoes from both clips landed within 5 cm of each other, and 5 of the 18 objects were linked across the clips: both
  shoes, the ladder (one clip only saw its feet) and two shirts, all right by eye. Together the videos must stay under
  250 frames after the stride.
- The `frames` stage keeps every `--stride`-th frame (stride 20 at 30 fps is 1.5 frames per second) and scales it
  down to 1280 pixels on the long side. A 60 s clip gives 90 frames.
- SAM is prompted with a fixed list of 60 indoor objects (`data/vocab.py`). `--vocab` takes another list, by name
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
`evaluate_openlex3d.py` (below) runs that protocol.

The third table scores the nearest-object query (`graph.query --near`) against the GT boxes (`metrics.nearest_objects`).
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

### OpenLex3D's protocol: comparing with published methods

`evaluate_openlex3d.py` scores a map with [OpenLex3D](https://github.com/openlex3d/openlex3d)'s own evaluation code. That
puts it next to the methods its paper scores on the same 8 scenes: ConceptGraphs, HOV-SG, OpenMask3D and Kassab2024.
The table lists the paper's own retrieval numbers (its Table 4: all queries, pooled over the scenes). OpenLex3D also
publishes the methods' predictions; scoring ConceptGraphs' here gives the paper's numbers to 0.01 (5.86, 11.32,
22.40 against 22.39), which checks that this script runs OpenLex3D's code as the authors did. `--baselines` is only
needed for that check, or for what the paper doesn't print, such as l0 and l1 separately.

```bash
uv sync --extra cu130 --extra openlex3d   # OpenLex3D's code, open3d and faiss
wget -P data/openlex3d_predictions http://aisdatasets.cs.uni-freiburg.de/openlex3d/predictions_sparse.zip   # 2 GB
unzip data/openlex3d_predictions/predictions_sparse.zip -d data/openlex3d_predictions/extracted
uv run --no-sync python evaluate_openlex3d.py --baselines concept-graphs   # check against the paper (optional)
uv run --no-sync python evaluate_openlex3d.py --pose-source vggt --variants graph memory tracks
uv run --no-sync python evaluate_openlex3d.py --table   # every method scored so far
```

**What gets scored.** Each map is exported to OpenLex3D's format in `outputs/openlex3d/predictions/<method>/replica/<scene>/`:

- each node as the graph holds it: its box, with the points its views agree on (`graph.multiview.object_points`) that
  lie inside the box, one per 2 cm cube;
- the object each point belongs to;
- each object's SigLIP 2 embedding (the mean over its masks, as `graph.query` searches).

Objects with no reliable mask have no points and are left out. `--variants` picks which objects:

- `graph`: the graph stage's nodes;
- `memory`: the memory's nodes before `graph/refine.py`;
- `tracks`: the raw SAM tracks, or the GT objects with `--mask-source gt`, which is an upper bound.

**Alignment.** With `--pose-source vggt`, the map is first moved into Replica's frame by the Sim(3) that best maps
VGGT-Omega's camera centres onto the GT ones. Scale is fitted too. The camera error left after the fit is saved with the
scores.

**The two tasks.** They run as OpenLex3D's scripts run them:

- **Segmentation, top-n:** every point of the GT cloud takes the n best of 1154 labels, from the nearest predicted point.
  Per object, the table gives the share of its points whose labels hold a synonym, a depiction, a visually similar label
  or clutter, or are wrong, or are missing (no predicted point within 5 cm).
- **Queries:** each OpenLex3D query takes its 10 best objects. Each object stands for the GT mesh vertices within 3.2 cm
  of its points, and is scored with instance-segmentation AP: mAP over IoU 0.5 to 0.95, AP50 and AP25, in percent.

Scores go to `outputs/openlex3d/results/<method>.json`.

**Caveats.**

- Our maps are read with SigLIP 2's text encoder, and the paper's predictions with CLIP ViT-H-14, as in the paper.
- Replica runs prompt SAM with the scene's own GT classes. Score `--vocab indoor` runs for a fair comparison.
- The paper's methods ran on every 10th frame (`--stride 10`).
- The 3.2 cm tolerance makes AP sensitive to depth and pose error, not only to the objects found.
- OpenLex3D's segmentation code takes about 2 minutes per scene for each top-n.

### Video

`views/video.py` renders a run's frames to an MP4 with each object's mask outlined and labelled with its class and ID, with
the SAM tracks and the graph nodes side by side. Run `run.py` first, since it reads the cached stages. A node takes the
colour of its largest track, so duplicates that the graph merged have two colours on the left and one on the right.

```bash
uv run --no-sync python -m views.video --source replica/room_0/00                            # eval/tracks_graph.mp4
uv run --no-sync python -m views.video --source replica/room_0/00 --panels tracks graph gt   # add the ground truth
```

### Graph: configuration, rebuilds and tuning

`configs/association.json` holds every matching parameter. Weights are normalised over the enabled cues
(`enable_dino`, `enable_geometry`), and a missing cue scores zero without redistributing its weight. Geometry is
`geometry_inside_weight * inside + (1 - geometry_inside_weight) * exp(-centre distance / center_distance_scale)`, where
inside is the share of the smaller box inside the larger (`mask_boxes.box_containment`). Not IoU: a view of part of an
object lies inside the object's box, which has grown to all its views, so IoU would score it near 0 and split the
object into many nodes. A match needs a score strictly above the threshold; in each frame the strongest pairs
are assigned first, one observation per object. `require_same_label` adds a same-label rule, off by default since
prompt labels change between views. `cut_gap`, `cut_dino`, `cut_dino_gap` (how far apart the boxes must also be for a
DINO cut; `null` lets DINO cut alone), `switch_margin` and `bad_frame_tracks` set when a track is cut;
`follow_tracks: false` matches every mask on its own instead, without merges. `cut_after: 2` cuts a track only where it
jumps in two reliable masks in a row, and keeps a one-frame jump in the node as a member only. That leaves fewer nodes
of one mask that flash up in the video (20 instead of 34 on IMG_0649), but a one-frame jump is mostly SAM's mask on
another object for that frame, so on Replica IDF1 falls from 0.598 to 0.588 and the nodes hold 1.28 GT objects each
instead of 1.15; it is off (1) by default.

Changing only the config recomputes only the graph. To rebuild it from the cached stages without loading any model
or scoring it:

```bash
uv run --no-sync python run.py --source replica/room_0/00 --graph-only --association-config configs/association.json
```

The graph stage directory holds the node per mask and each node's box (`arrays.npz`, read by the evaluation, video
and plots), `memory/`
(the compact object memory: IDs, labels, boxes, DINO descriptors, SigLIP sums, first and last frame;
`graph.build.load_memory()` restores it and can go on with later frames), and `matches.json` and `history.json` (each
observation's route, `track`, `match` or `new`, with its cut and scores; the node count per frame), and `merges.json`
(each merge with the IDs of the time and the final `node`), for inspection. `--skip-text-eval` scores
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
