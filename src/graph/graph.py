"""Merge SAM tracks into object nodes: the graph stage.

A SAM track is every mask with one SAM ID. SAM 3 splits one object over several tracks in three ways:
  duplicates  masks of different prompts on one object in the same frames (a sofa is also found as "cushion")
  re-entries  the object leaves the view and comes back under a new ID
  videos      a run of several videos tracks each on its own (sam.segment_clips), so each video has its own IDs

Replica objects never move, so two tracks seen in the same frames say a lot about each other: in how many of those
frames do their masks overlap (IoU >= OVERLAP_IOU)?
  most of them (DUPLICATE_SHARE)   duplicates, which pass 1 merges
  hardly any (APART_SHARE)         two objects, which never end up in one node, not even through other tracks
  in between                       a SAM ID that drifted onto another object for a while: pass 2 decides
Pass 2 merges the remaining tracks that have the same label (the prompt with the most SAM score) and whose 3D boxes
overlap (BOX_IOU), or where one box lies mostly inside the other (CONTAIN): a view of only part of the object, such as
a ladder's feet. It is greedy: it merges the pair whose boxes overlap most, recomputes that node's box, and repeats.

DINO can also veto merges whose mean embeddings differ (DINO_SIM), but it is off. On Replica it changed nothing, and
on two phone videos of one room it blocked correct links between the videos: the same shoe or shirt seen from another
side scored 0.45-0.58. The same-label rule took its place and kept the Replica scores.
"""

from collections import defaultdict

import numpy as np
import pycocotools.mask as mask_utils

from graph.node import fused_box, mean_embedding
from utils.schema import Boxes, Features, Graph, Masks

OVERLAP_IOU = 0.5  # mask IoU at which two masks in a frame are one object
DUPLICATE_SHARE = 0.5  # share of the shared frames with overlapping masks from which two tracks are duplicates
APART_SHARE = 0.1  # and up to which they are two objects
BOX_IOU = 0.2  # 3D IoU of the padded boxes needed to merge in pass 2
BOX_PAD = 0.05  # metres added to each side of a box, so thin objects (pictures, rugs) have a volume
CONTAIN = 0.7  # share of the smaller padded box inside the larger at which pass 2 also merges
DINO_SIM = None  # cosine similarity of the mean DINO embeddings also needed in pass 2; None turns the veto off

PARAMS = {
    "overlap_iou": OVERLAP_IOU,
    "duplicate_share": DUPLICATE_SHARE,
    "apart_share": APART_SHARE,
    "box_iou": BOX_IOU,
    "box_pad": BOX_PAD,
    "contain": CONTAIN,
    "same_label": True,
    "dino_sim": DINO_SIM,
    "node_box": "points, outlier masks dropped",  # graph.node.fused_box
}


def build(masks: Masks, boxes: Boxes, features: Features | None = None, trace: list | None = None) -> Graph:
    """Node of every mask. features are only needed when DINO_SIM turns the DINO veto on.

    trace, when given, gets every merge in order as (rule, SAM ID, SAM ID, value): ("duplicate", a, b, share of
    shared frames with overlapping masks) from pass 1, ("box overlap", a, b, 3D IoU) or ("inside", a, b, share of the
    smaller box inside the other) from pass 2. a and b stand for the clusters merged, by their first track.
    """
    tracks, track_of = np.unique(masks.sam_id, return_inverse=True)
    track_rows = [np.flatnonzero(track_of == t) for t in range(len(tracks))]
    shared = overlap_share(masks, track_of)

    # Clusters of tracks, indexed by their first track; alive marks the ones not merged into another
    members = [[t] for t in range(len(tracks))]
    alive = np.ones(len(tracks), bool)
    apart = np.zeros((len(tracks), len(tracks)), bool)  # seen together as different objects: never merge
    for (a, b), share in shared.items():
        apart[a, b] = apart[b, a] = share <= APART_SHARE
    cluster = np.arange(len(tracks))  # cluster of each track

    def merge(i, j):
        members[i] += members[j]
        cluster[members[j]] = i
        alive[j] = False
        apart[i] |= apart[j]
        apart[:, i] |= apart[:, j]

    # Pass 1: duplicates, the best overlap first
    for (a, b), share in sorted(shared.items(), key=lambda kv: -kv[1]):
        i, j = sorted((cluster[a], cluster[b]))
        if share >= DUPLICATE_SHARE and i != j and not apart[i, j]:
            if trace is not None:
                trace.append(("duplicate", int(tracks[i]), int(tracks[j]), share))
            merge(i, j)

    # Pass 2: re-entries, drifted tracks and other videos, on cluster-level labels, boxes and embeddings
    veto = DINO_SIM is not None and features is not None
    lo, hi = np.full((len(tracks), 3), np.nan, np.float32), np.full((len(tracks), 3), np.nan, np.float32)
    emb = np.zeros((len(tracks), features.dino.shape[1] if veto else 1), np.float32)
    label = np.zeros(len(tracks), int)

    def update(i):
        rows = np.concatenate([track_rows[t] for t in members[i]])
        lo[i], hi[i] = fused_box(boxes, rows)
        label[i] = np.bincount(masks.prompt[rows], weights=masks.score[rows]).argmax()
        if veto:
            emb[i] = mean_embedding(features, rows)

    for i in np.flatnonzero(alive):
        update(i)
    while True:
        iou = padded_box_iou(lo, hi)
        inside = np.nan_to_num(padded_box_containment(lo, hi), nan=0)
        inside = np.maximum(inside, inside.T)
        geometry = (iou >= BOX_IOU) | (inside >= CONTAIN)  # NaN boxes compare False
        ok = alive[:, None] & alive[None] & ~apart & geometry & (label[:, None] == label[None])
        np.fill_diagonal(ok, False)
        if veto:  # appearance only vetoes: where an object is decides which merge comes first
            ok &= emb @ emb.T >= DINO_SIM
        if not ok.any():
            break
        # Overlapping boxes first, then the ones that only contain each other
        order = np.where(iou >= BOX_IOU, 1 + np.nan_to_num(iou), inside)
        i, j = sorted(np.unravel_index(np.where(ok, order, -np.inf).argmax(), ok.shape))
        if trace is not None:
            rule = ("box overlap", float(iou[i, j])) if iou[i, j] >= BOX_IOU else ("inside", float(inside[i, j]))
            trace.append((rule[0], int(tracks[i]), int(tracks[j]), rule[1]))
        merge(i, j)
        update(i)

    # Number the nodes 0, 1, ... in order of first appearance
    node_of_cluster = {}
    for c in cluster[track_of[np.argsort(masks.frame, kind="stable")]]:
        node_of_cluster.setdefault(int(c), len(node_of_cluster))
    return Graph(node=np.array([node_of_cluster[int(c)] for c in cluster[track_of]], np.int32))


def overlap_share(masks: Masks, track_of: np.ndarray) -> dict[tuple[int, int], float]:
    """For every pair of tracks (a < b) seen in the same frames: the share of those frames where their masks overlap."""
    overlaps = defaultdict(list)
    for frame in np.unique(masks.frame):
        rows = np.flatnonzero(masks.frame == frame)
        rles = [masks.rle[r] for r in rows]
        iou = mask_utils.iou(rles, rles, [0] * len(rles))
        for i, j in zip(*np.triu_indices(len(rows), 1), strict=True):
            a, b = sorted((int(track_of[rows[i]]), int(track_of[rows[j]])))
            if a != b:
                overlaps[a, b].append(iou[i, j] >= OVERLAP_IOU)
    return {pair: float(np.mean(v)) for pair, v in overlaps.items()}


def padded_box_containment(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """(M, M) share of box i's volume inside box j, both grown by BOX_PAD on every side. NaN where a box is NaN."""
    lo, hi = lo - BOX_PAD, hi + BOX_PAD
    inter = np.prod(np.clip(np.minimum(hi[:, None], hi[None]) - np.maximum(lo[:, None], lo[None]), 0, None), axis=2)
    return inter / np.prod(hi - lo, axis=1)[:, None]


def padded_box_iou(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """(M, M) 3D IoU of every pair of boxes, each grown by BOX_PAD on every side. NaN where a box is NaN."""
    lo, hi = lo - BOX_PAD, hi + BOX_PAD
    inter = np.prod(np.clip(np.minimum(hi[:, None], hi[None]) - np.maximum(lo[:, None], lo[None]), 0, None), axis=2)
    volume = np.prod(hi - lo, axis=1)
    return inter / (volume[:, None] + volume[None] - inter)
