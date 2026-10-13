"""Whole-video fixes to the object memory's nodes, once every frame is in: split, label, merge, and keep the sure ones.

The scene is static, and the whole video is in. The memory (graph/association.py) decides one frame at a time; seen
whole, some of its nodes are wrong in ways only 3D shows. Like ConceptGraphs' post-processing (Gu et al., 2023: DBSCAN
denoising, merging objects whose points overlap, dropping objects seen in under 3 detections), this keeps fewer
objects rather than more, and only those whose box can be trusted:

  split      a node whose points form clusters apart: two monitors a SAM track slid between. The points of the node's
             reliable masks are binned in cubes of VOXEL metres. A cube holding at least DENSITY of the node's typical
             cube (the one its average point is in) is dense, so a few points between two objects don't join them, and
             dense cubes within SPLIT_GAP of each other make a cluster. Each reliable mask goes with the cluster
             holding most of its points, and each cluster's masks become a node, the one with the most masks keeping
             the ID. The node's other masks follow the nearest in time of those of their SAM track, else of the node's.
  label      each node's label from all its views (labels): SAM's votes for its prompts, weighed by how well SigLIP's
             mean embedding of the node matches each prompt's text.
  duplicate  two nodes of one label that are two masks of one object in the frames they share, such as a chair and
             its seat (duplicate_merge): the smaller one's mask lies inside the other's.
  merge      two nodes of one label, alike in DINO (cosine of their mean descriptors at least MERGE_DINO), where at
             least CONTAIN of the box of the one seen in fewer masks lies inside the other's, and at least OVERLAP of
             its points are within NEAR of the other's: the parts of one object the memory kept apart, such as a sofa
             seen from one end and later from the other. Containment rather than IoU, since a part's box is much
             smaller than the whole's; the points are ConceptGraphs' overlap, and with the part being the node seen
             less, they keep a box that few views drew, stretched over its neighbours, from taking them all. The whole
             keeps its box, so a merge can't grow a box into the next object. The points decide: the box test is
             loose because a thin object's box is as deep as VGGT-Omega's depth is unsure, a monitor's 0.1 to 0.45 m.
             Nodes seen in one frame as two masks merge neither here nor below, unless that is a slip of SAM's tracker:
             at most TOGETHER of the frames of the one seen in fewer (seen_apart).
  agree      two nodes of one label that the frames seeing both agree are one (consensus_merge, as MaskClustering's
             view consensus): parts whose boxes don't nest, seen from different sides.
  keep       only the nodes whose box can be trusted (drop): at least MIN_MASKS masks and MIN_VIEWS reliable ones,
             views spanning at least MIN_ARC degrees around the object, at least MIN_COVER coverage (quality: in its
             frames, its points in view lie inside its own mask, as one static object's do), and not only ever small
             and far. The others' masks get node -1.

On the 8 Replica scenes with VGGT-Omega's depth, a scene keeps 46 nodes of the memory's 121, and 7 of them match no GT
object instead of 48; 5% of the object boxes kept are off (IoU under 0.2), against 21%, and their mean IoU is 0.54
instead of 0.45. Of the 368 revisits (an object leaves the view and comes back), the merges keep the object's node in
224 and switch it in 34, against 214 and 47 for the memory alone; after the last step 203 keep it and 17 switch, and
38 more fall on nodes that were dropped. The price is the objects whose box couldn't be checked: 18.5 objects a scene
get a good box (IoU 0.5 or more) instead of 25. Checked alone against Replica's objects, 41 of 42 box merges, all 14
consensus merges between objects and 89% of the duplicate merges joined one object. refine returns the node of every
mask after all of this, the nodes' labels, and what each step did, for the graph stage's refine.json.
"""

import warnings
from collections import Counter, defaultdict

import numpy as np
import pycocotools.mask as mask_utils
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from graph import shape
from graph.node import mask_points
from utils import lift3d
from utils.schema import Boxes, Features, Masks, Run, TextAligned

VOXEL = 0.1  # metres: side of the cubes a node's points are clustered in
DENSITY = 0.1  # share of the node's typical cube (the one its average point is in) a cube must hold to be in a cluster
SPLIT_GAP = 0.3  # metres between the centres of two clusters' nearest cubes for them to be apart
CONTAIN = 0.2  # share of the box of the node seen in fewer masks inside the other's at which the two may merge
PAD = 0.05  # metres both boxes are grown by on each side before containment, so that flat things can overlap
MERGE_DINO = 0.7  # cosine of two nodes' mean DINO descriptors under which they don't merge
TOGETHER = 0.2  # share of the smaller node's frames two nodes may share as two masks and still merge
NEAR = 0.15  # metres from the whole's points within which a part's point is on the whole
OVERLAP = 0.5  # share of the part's points that must be on the whole for the two to merge
VISIBLE = 0.5  # share of a node's points in a frame's image and not hidden for the frame to see the node
CONTAINED = 0.8  # share of a node's points in view that a mask must hold to hold the node (as MaskClustering's)
CONSENSUS = 0.8  # share of the frames seeing two nodes in which one mask holds both, for them to merge
MIN_OBSERVERS = 3  # frames that must see both nodes, with one of them in a mask, before their consensus counts
LABEL_SCALE = 100.0  # SigLIP cosines are multiplied by this in the softmax that weighs SAM's votes for a label
DUPLICATE = 0.9  # share of the smaller node's mask inside the other's, median over frames with both, to be one object
MIN_SHARED = 2  # frames two nodes must share before they count as duplicates
DUPLICATE_NEST = 0.5  # share of the smaller duplicate's box (graph/shape.py's) that must lie inside the other's
SIGHTING_POINTS = 512  # of a node's points projected into every frame for its sightings
MIN_MASKS = 3  # nodes with fewer masks are dropped, as ConceptGraphs drops objects in under 3 detections
MIN_COVER = 0.7  # mean share of a node's points in view that lie inside its own mask, over its frames
MIN_VIEWS = 10  # reliable masks a node needs for its box to be trusted
MIN_ARC = 15.0  # degrees around the object (about the vertical) its reliable views must span
SMALL = 0.002  # share of the image under which a mask is small
FAR = 4.0  # metres from the camera beyond which a mask is far
PARAMS = {
    "voxel": VOXEL,
    "density": DENSITY,
    "split_gap": SPLIT_GAP,
    "contain": CONTAIN,
    "pad": PAD,
    "merge_dino": MERGE_DINO,
    "together": TOGETHER,
    "near": NEAR,
    "overlap": OVERLAP,
    "visible": VISIBLE,
    "contained": CONTAINED,
    "consensus": CONSENSUS,
    "min_observers": MIN_OBSERVERS,
    "label_scale": LABEL_SCALE,
    "duplicate": DUPLICATE,
    "min_shared": MIN_SHARED,
    "duplicate_nest": DUPLICATE_NEST,
    "min_masks": MIN_MASKS,
    "min_cover": MIN_COVER,
    "min_views": MIN_VIEWS,
    "min_arc": MIN_ARC,
    "small": SMALL,
    "far": FAR,
}


def clusters(points: np.ndarray, voxel: float = VOXEL, density: float = DENSITY, gap: float = SPLIT_GAP) -> np.ndarray:
    """(k,) cluster of each of (k, 3) points: cubes of side voxel dense enough (DENSITY) and within gap of each other.

    Points in sparser cubes get -1, so that a few points between two objects don't join them.
    """
    cube = np.floor(points / voxel).astype(np.int64)
    keys, inverse, counts = np.unique(cube, axis=0, return_inverse=True, return_counts=True)
    typical = (counts**2).sum() / counts.sum()  # points in the cube of the average point
    dense = np.flatnonzero(counts >= max(2, density * typical))
    a, b = cKDTree(keys[dense] * voxel).query_pairs(gap + 1e-6, output_type="ndarray").reshape(-1, 2).T
    _, part = connected_components(csr_matrix((np.ones(len(a)), (a, b)), (len(dense), len(dense))), directed=False)
    cluster = np.full(len(keys), -1)
    cluster[dense] = part
    return cluster[inverse.reshape(-1)]


def split(
    masks: Masks, boxes: Boxes, node: np.ndarray, voxel: float = VOXEL, density: float = DENSITY, gap: float = SPLIT_GAP
) -> tuple[np.ndarray, list[dict]]:
    """The node of every mask, with each node whose points form clusters apart split, and a record of each split."""
    node = node.copy()
    trusted = lift3d.reliable(boxes)
    next_id = int(node.max()) + 1 if len(node) else 0
    records = []
    for n in np.unique(node[node >= 0]).tolist():
        rows = np.flatnonzero(node == n)
        good = rows[trusted[rows]]
        if len(good) < 2 or boxes.points is None:
            continue
        points = boxes.points[good].astype(np.float32).reshape(-1, 3)
        owner = np.repeat(np.arange(len(good)), boxes.points.shape[1])
        valid = np.isfinite(points).all(axis=1)
        cluster = clusters(points[valid], voxel, density, gap)
        owner, cluster = owner[valid][cluster >= 0], cluster[cluster >= 0]
        part = np.full(len(rows), -1)
        for k, r in enumerate(good.tolist()):
            if np.any(owner == k):
                part[rows == r] = np.bincount(cluster[owner == k]).argmax()
        if len(np.unique(part[part >= 0])) < 2:
            continue
        placed = rows[part >= 0]
        for i in np.flatnonzero(part < 0).tolist():
            same = placed[masks.sam_id[placed] == masks.sam_id[rows[i]]]
            pool = same if len(same) else placed
            part[i] = part[rows == pool[np.argmin(np.abs(masks.frame[pool] - masks.frame[rows[i]]))]][0]
        sizes = np.bincount(part)
        ids = [n]
        for k in np.argsort(-sizes, kind="stable")[1 : np.count_nonzero(sizes)].tolist():
            node[rows[part == k]] = next_id
            ids.append(next_id)
            next_id += 1
        records.append({"node": n, "parts": ids, "masks": [int((node[rows] == i).sum()) for i in ids]})
    return node, records


def spread(points: np.ndarray, k: int) -> np.ndarray:
    """At most k of (n, 3) points, spread over them."""
    return points[np.unique(np.linspace(0, len(points) - 1, min(len(points), k)).astype(int))]


def node_points(boxes: Boxes, rows: np.ndarray) -> np.ndarray:
    """(k, 3) sampled points of a node's reliable masks, or of all its masks with points when none is reliable."""
    good = rows[lift3d.reliable(boxes)[rows]]
    points = mask_points(boxes, good if len(good) else rows[boxes.n_points[rows] > 0])
    return points if len(points) else np.full((1, 3), np.inf, np.float32)


def labels(
    masks: Masks, node: np.ndarray, emb: TextAligned | None = None, text: np.ndarray | None = None, scale=LABEL_SCALE
) -> dict[int, int]:
    """Each node's label by ID, an index into the prompts: the one with the most SAM score summed over its masks.

    Given the masks' SigLIP embeddings (emb) and the prompts' text embeddings (text, (V, D), as SigLIP encodes them),
    each prompt's votes are weighed by softmax(scale * cosine) of its text against the node's mean embedding, over all
    its views. On Replica that names the GT object's class for 72.5% of the nodes, against 70.3% for the votes alone
    and 55% for SigLIP alone: SAM's prompts propose, and SigLIP decides between those that are close.
    """
    out = {}
    for n in np.unique(node[node >= 0]).tolist():
        rows = node == n
        n_prompts = len(text) if text is not None else 0
        votes = np.bincount(masks.prompt[rows], weights=masks.score[rows], minlength=n_prompts)
        if emb is not None and text is not None:
            mean = emb.siglip[rows].astype(np.float32).sum(axis=0)
            cosine = text @ (mean / max(np.linalg.norm(mean), 1e-9))
            votes = votes * np.exp(scale * (cosine - cosine.max()))
        out[n] = int(votes.argmax())
    return out


def seen_apart(a: set[int], b: set[int], together: float = TOGETHER) -> bool:
    """Whether two nodes seen in frames a and b are two objects: they share more than together of the smaller's frames.

    Two masks in one frame are two objects, unless one of them is a slip of SAM's tracker for a frame or two.
    """
    return len(a & b) > together * min(len(a), len(b))


def inside(part: tuple[np.ndarray, np.ndarray], whole: tuple[np.ndarray, np.ndarray]) -> float:
    """Share of box part ((min, max)) that lies inside box whole."""
    inter = np.prod(np.clip(np.minimum(part[1], whole[1]) - np.maximum(part[0], whole[0]), 0, None))
    return float(inter / max(np.prod(part[1] - part[0]), 1e-9))


def merge(
    masks: Masks,
    boxes: Boxes,
    features: Features,
    node: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
    label: dict[int, int],
    apart: list[list[int]] = (),
    contain: float = CONTAIN,
    pad: float = PAD,
    merge_dino: float = MERGE_DINO,
    near: float = NEAR,
    overlap: float = OVERLAP,
    together: float = TOGETHER,
) -> tuple[np.ndarray, list[dict]]:
    """The node of every mask, with nodes of one look and label merged where one's box lies mostly inside the other's.

    Of two nodes, the one seen in fewer masks must lie inside the other, which takes it: a part is seen less than the
    whole, and a box that few views drew can be stretched over its neighbours, which would then all lie inside it.
    box_min and box_max are by node ID, label the nodes' labels (labels). apart lists groups of nodes never to merge
    with each other: the parts of a split. Two nodes seen in one frame as two masks merge only when they share at most
    together of the frames of the one seen in fewer (seen_apart).
    """
    node = node.copy()
    box = {n: (box_min[n] - pad, box_max[n] + pad) for n in label if np.isfinite(box_min[n]).all()}
    frames = {n: set(masks.frame[node == n].tolist()) for n in box}
    dino = {n: features.dino[node == n].astype(np.float32).sum(axis=0) for n in box}  # unit descriptors summed
    count = {n: int((node == n).sum()) for n in box}
    volume = {n: float(np.prod(box[n][1] - box[n][0])) for n in box}
    points = {n: node_points(boxes, np.flatnonzero(node == n)) for n in box}
    trees = {}
    group = {n: k for k, ids in enumerate(apart) for n in ids}  # node -> its split
    records = []
    while True:
        best = None
        ids = sorted(box)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if label[a] != label[b] or seen_apart(frames[a], frames[b], together):
                    continue
                if a in group and group[a] == group.get(b):
                    continue
                whole, part = (a, b) if (count[a], volume[a]) >= (count[b], volume[b]) else (b, a)
                c = inside(box[part], box[whole])
                if c < contain or (best is not None and c <= best[0]):
                    continue
                similarity = dino[a] @ dino[b] / max(np.linalg.norm(dino[a]) * np.linalg.norm(dino[b]), 1e-9)
                if similarity < merge_dino:
                    continue
                if whole not in trees:
                    trees[whole] = cKDTree(points[whole])
                sample = spread(points[part], 2048)
                on = float(np.mean(trees[whole].query(sample, distance_upper_bound=near)[0] <= near))
                if on >= overlap:
                    best = (c, whole, part, similarity, on)
        if best is None:
            return node, records
        c, keep, gone, similarity, on = best
        node[node == gone] = keep
        count[keep] += count.pop(gone)
        points[keep] = np.concatenate([points[keep], points.pop(gone)])
        trees.pop(keep, None)
        trees.pop(gone, None)
        del volume[gone]
        frames[keep] |= frames.pop(gone)
        dino[keep] = dino[keep] + dino.pop(gone)
        del box[gone]
        if gone in group:
            group.setdefault(keep, group[gone])
        scores = {"containment": round(c, 3), "dino": round(float(similarity), 3), "overlap": round(on, 3)}
        records.append({"kept": keep, "merged": gone, **scores})


def project(points: np.ndarray, depth: np.ndarray, K: np.ndarray, c2w: np.ndarray):
    """Pixels u and v of (k, 3) points in a view, and whether each is seen: in the image and not hidden.

    Hidden is further behind the view's depth there than graph/shape.py's HIDDEN. Points not seen get pixel (0, 0).
    """
    h, w = depth.shape
    w2c = np.linalg.inv(c2w)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = np.maximum(cam[:, 2], 1e-6)
    u = np.round(K[0, 0] * cam[:, 0] / z + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * cam[:, 1] / z + K[1, 2]).astype(int)
    seen = (cam[:, 2] > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v = np.where(seen, u, 0), np.where(seen, v, 0)
    there = depth[v, u]
    seen &= cam[:, 2] <= there + np.maximum(shape.HIDDEN[0], shape.HIDDEN[1] * there)
    return u, v, seen


def quality(run: Run, masks: Masks, boxes: Boxes, node: np.ndarray) -> dict[int, dict]:
    """How far each node's box can be trusted, by node ID: its coverage, views and arc.

    coverage  the mean, over the node's frames that see at least 20 of its points, of the share of those points that
              lie inside its own mask there. The scene is static, so one object's points seen in a frame are inside
              its mask, and a node that joined masks of several objects (a far monitor and a table top SAM called a
              monitor) has the others' points outside. None when no frame sees enough of it.
    views     its reliable masks (lift3d.reliable)
    arc       degrees about the vertical that the directions from those views' cameras to the node span: a node seen
              from one direction has a depth extent no other view checked.
    """
    trusted = lift3d.reliable(boxes)
    get = shape.view_source(run)
    ids = np.unique(node[node >= 0]).tolist()
    points = {n: spread(node_points(boxes, np.flatnonzero(node == n)), SIGHTING_POINTS) for n in ids}
    covers = {n: [] for n in ids}
    centres = {}
    for frame in np.unique(masks.frame[node >= 0]).tolist():
        rows = np.flatnonzero((masks.frame == frame) & (node >= 0))
        depth, K, c2w = get(frame)
        centres[frame] = c2w[:3, 3]
        stack = mask_utils.decode([masks.rle[r] for r in rows]).astype(bool)
        for n in np.unique(node[rows]).tolist():
            if not np.isfinite(points[n]).all():
                continue
            u, v, seen = project(points[n], depth, K, c2w)
            if seen.sum() >= 20:
                mask = shape.shrink(stack[:, :, node[rows] == n].any(axis=2), depth.shape)
                covers[n].append(float(mask[v[seen], u[seen]].mean()))
    out = {}
    for n in ids:
        good = np.flatnonzero((node == n) & trusted)
        arc = 0.0
        if len(good) >= 2 and np.isfinite(points[n]).all():
            rays = np.median(points[n], axis=0) - np.stack([centres[int(f)] for f in masks.frame[good]])
            azimuth = np.sort(np.degrees(np.arctan2(rays[:, 1], rays[:, 0])))
            arc = float(360 - np.diff(np.r_[azimuth, azimuth[0] + 360]).max())  # the circle less its widest gap
        coverage = float(np.mean(covers[n])) if covers[n] else None
        out[n] = {"coverage": coverage, "views": len(good), "arc": round(arc, 1)}
    return out


def sightings(
    run: Run, masks: Masks, points: dict[int, np.ndarray], visible: float = VISIBLE, contained: float = CONTAINED
) -> dict[int, dict[int, set[int]]]:
    """By frame, the nodes it sees, each with the rows of the frame's masks that hold it.

    points holds each node's points by ID. A frame sees a node when at least visible of its points are in the image
    and not hidden (as in graph/shape.py), and a mask holds it when at least contained of those lie inside the mask.
    """
    get = shape.view_source(run)
    ids = list(points)
    stacked = np.concatenate([points[n] for n in ids])
    owner = np.repeat(np.arange(len(ids)), [len(points[n]) for n in ids])
    total = np.bincount(owner, minlength=len(ids))
    out = {}
    for frame in np.unique(masks.frame).tolist():
        depth, K, c2w = get(frame)
        h, w = depth.shape
        u, v, seen = project(stacked, depth, K, c2w)
        in_view = np.flatnonzero(np.bincount(owner[seen], minlength=len(ids)) >= np.maximum(visible * total, 1))
        if not len(in_view):
            continue
        rows = np.flatnonzero(masks.frame == frame)
        stack = mask_utils.decode([masks.rle[r] for r in rows]).astype(bool)
        small = [shape.shrink(stack[:, :, k], (h, w)) for k in range(len(rows))]
        view = {}
        for i in in_view.tolist():
            pick = seen & (owner == i)
            view[ids[i]] = {int(r) for k, r in enumerate(rows) if small[k][v[pick], u[pick]].mean() >= contained}
        out[frame] = view
    return out


def duplicate_merge(
    masks: Masks,
    node: np.ndarray,
    label: dict[int, int],
    apart: list[list[int]] = (),
    duplicate: float = DUPLICATE,
    min_shared: int = MIN_SHARED,
    any_label: bool = False,
    boxes3d: tuple[np.ndarray, np.ndarray] | None = None,
    nest: float = 0.0,
) -> tuple[np.ndarray, list[dict]]:
    """The node of every mask, with nodes that are two masks of one object in the frames they share merged.

    SAM gives some objects two masks in a frame, such as a chair and its seat, overlapping too little to count as one
    observation (the memory's duplicate_iou). Each follows its own track into its own node, and the two are seen in
    one frame, which the other merges take as two objects. Here two nodes of one label (any label with any_label) that
    share at least min_shared frames merge when, in the median frame they share, at least duplicate of the smaller
    one's mask lies inside the other's. Pairs go in order of that share, never joining the parts of a split; the node
    of more masks takes the other.
    """
    node = node.copy()
    shares = defaultdict(list)
    for frame in np.unique(masks.frame).tolist():
        rows = np.flatnonzero((masks.frame == frame) & (node >= 0))
        ids = np.unique(node[rows]).tolist()
        if len(ids) < 2:
            continue
        rles = [mask_utils.merge([masks.rle[r] for r in rows[node[rows] == n]]) for n in ids]
        area = mask_utils.area(rles).astype(float)
        iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(ids)))
        for i, a in enumerate(ids):
            for j in range(i + 1, len(ids)):
                if any_label or label[a] == label[ids[j]]:
                    inter = iou[i, j] * (area[i] + area[j]) / (1 + iou[i, j])
                    shares[a, ids[j]].append(inter / max(min(area[i], area[j]), 1.0))
    rates = sorted(((float(np.median(v)), k) for k, v in shares.items() if len(v) >= min_shared), reverse=True)
    members = {n: {n} for n in label}
    count = {n: int((node == n).sum()) for n in label}
    root = {n: n for n in label}
    records = []
    for share, (a, b) in rates:
        if share < duplicate:
            break
        ra, rb = root[a], root[b]
        if ra == rb or any(members[ra] & set(g) and members[rb] & set(g) for g in apart):
            continue
        if boxes3d is not None:
            box_a, box_b = (boxes3d[0][a], boxes3d[1][a]), (boxes3d[0][b], boxes3d[1][b])
            small, large = sorted((box_a, box_b), key=lambda box: np.prod(box[1] - box[0]))
            if not np.isfinite(small[0]).all() or not np.isfinite(large[0]).all() or inside(small, large) < nest:
                continue
        keep, gone = (ra, rb) if count[ra] >= count[rb] else (rb, ra)
        for n in members[gone]:
            root[n] = keep
        members[keep] |= members.pop(gone)
        count[keep] += count.pop(gone)
        node[node == gone] = keep
        records.append({"kept": keep, "merged": gone, "inside": round(share, 3), "frames": len(shares[a, b])})
    return node, records


def consensus_merge(
    masks: Masks,
    node: np.ndarray,
    label: dict[int, int],
    seen: dict[int, dict[int, set[int]]],
    apart: list[list[int]] = (),
    consensus: float = CONSENSUS,
    min_observers: int = MIN_OBSERVERS,
    together: float = TOGETHER,
) -> tuple[np.ndarray, list[dict]]:
    """The node of every mask, with nodes of one label merged where the frames that see both agree they are one.

    As MaskClustering's view consensus (Yan et al., 2024): a frame that sees two nodes (seen, from sightings), with
    at least one of them held by a mask of their label, observes them; it supports them when one such mask holds both.
    Nodes whose observers number at least min_observers, and at least consensus of them support, merge: the parts of
    one object that no single frame joined, seen from sides whose boxes don't nest. Merges go in order of consensus,
    never join nodes seen apart (seen_apart), nor the parts of a split; the node of more masks takes the other.
    """
    node = node.copy()
    observers, support = Counter(), Counter()
    for view in seen.values():
        held = {n: {r for r in rows if masks.prompt[r] == label[n]} for n, rows in view.items() if n in label}
        ids = sorted(held)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if label[a] != label[b] or not (held[a] or held[b]):
                    continue
                observers[a, b] += 1
                support[a, b] += bool(held[a] & held[b])
    rates = sorted(
        ((support[k] / n, k) for k, n in observers.items() if n >= min_observers and support[k] / n >= consensus),
        reverse=True,
    )
    members = {n: {n} for n in label}
    frames = {n: set(masks.frame[node == n].tolist()) for n in label}
    count = {n: int((node == n).sum()) for n in label}
    root = {n: n for n in label}
    records = []
    for rate, (a, b) in rates:
        ra, rb = root[a], root[b]
        if ra == rb or seen_apart(frames[ra], frames[rb], together):
            continue
        if any(members[ra] & set(g) and members[rb] & set(g) for g in apart):
            continue
        keep, gone = (ra, rb) if count[ra] >= count[rb] else (rb, ra)
        for n in members[gone]:
            root[n] = keep
        members[keep] |= members.pop(gone)
        frames[keep] |= frames.pop(gone)
        count[keep] += count.pop(gone)
        node[node == gone] = keep
        records.append({"kept": keep, "merged": gone, "consensus": round(rate, 3), "observers": observers[a, b]})
    return node, records


def camera_centres(run: Run, frames) -> dict[int, np.ndarray]:
    """The camera's position in the world (metres) in each of frames."""
    get = shape.view_source(run)
    return {int(f): get(int(f))[2][:3, 3] for f in np.unique(frames).tolist()}


def drop(
    run: Run,
    masks: Masks,
    boxes: Boxes,
    node: np.ndarray,
    min_masks: int = MIN_MASKS,
    small: float = SMALL,
    far: float = FAR,
    min_cover: float = MIN_COVER,
    min_views: int = MIN_VIEWS,
    min_arc: float = MIN_ARC,
) -> tuple[np.ndarray, list[dict]]:
    """The node of every mask, with -1 for the nodes whose box isn't to be trusted.

    Those are the nodes of fewer than min_masks masks, whose every mask is small and far, or that fall short of
    min_cover, min_views or min_arc (quality). The records give each dropped node's masks, the reasons, the nearest any
    of its masks was in metres (None without points), and its quality.
    """
    node = node.copy()
    if not len(node):
        return node, []
    height, width = masks.rle[0]["size"]
    area = lift3d.rle_area(masks.rle) / (height * width)
    distance = np.full(len(node), np.inf)
    if boxes.points is not None:
        centre = camera_centres(run, masks.frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # masks without points: all NaN
            median = np.nanmedian(boxes.points.astype(np.float32), axis=1)
        camera = np.stack([centre[int(f)] for f in masks.frame])
        distance = np.where(np.isfinite(median).all(axis=1), np.linalg.norm(median - camera, axis=1), np.inf)
    tiny = (area < small) & ~(distance <= far)
    trust = quality(run, masks, boxes, node)
    records = []
    for n in np.unique(node[node >= 0]).tolist():
        rows = node == n
        q = trust[n]
        reasons = [
            reason
            for reason, short in (
                ("few masks", int(rows.sum()) < min_masks),
                ("small and far", bool(tiny[rows].all())),
                ("inconsistent", q["coverage"] is None or q["coverage"] < min_cover),
                ("few views", q["views"] < min_views),
                ("narrow views", q["arc"] < min_arc),
            )
            if short
        ]
        if reasons:
            node[rows] = -1
            nearest = float(distance[rows].min()) if np.isfinite(distance[rows]).any() else None
            records.append({"node": n, "masks": int(rows.sum()), "reasons": reasons, "nearest": nearest, **q})
    return node, records


def refine(
    run: Run,
    masks: Masks,
    boxes: Boxes,
    features: Features,
    emb: TextAligned,
    text: np.ndarray | None,
    node: np.ndarray,
    **params,
) -> tuple[np.ndarray, dict[int, int], dict]:
    """The node of every mask after all the steps (-1 for dropped), the nodes' labels, and what each step did.

    emb and text are the masks' and the prompts' SigLIP embeddings for the labels (labels; text None: SAM's votes
    alone). params override the module's settings by PARAMS key.
    The node IDs are those of node, with new ones after its largest for the parts of splits, and gaps where nodes merged
    or were dropped.
    """
    p = {**PARAMS, **params}
    node, splits = split(masks, boxes, node, p["voxel"], p["density"], p["split_gap"])
    lo, hi = shape.object_boxes(run, masks, boxes, node)
    label = labels(masks, node, emb, text, p["label_scale"])
    apart = [s["parts"] for s in splits]
    node, duplicates = duplicate_merge(
        masks, node, label, apart, p["duplicate"], p["min_shared"], boxes3d=(lo, hi), nest=p["duplicate_nest"]
    )
    label = {n: label[n] for n in np.unique(node[node >= 0]).tolist()}
    keys = ("contain", "pad", "merge_dino", "near", "overlap", "together")
    node, merges = merge(masks, boxes, features, node, lo, hi, label, apart, **{k: p[k] for k in keys})
    label = {n: label[n] for n in np.unique(node[node >= 0]).tolist()}
    points = {n: spread(node_points(boxes, np.flatnonzero(node == n)), SIGHTING_POINTS) for n in label}
    points = {n: x for n, x in points.items() if np.isfinite(x).all()}  # nodes without depth can't be seen
    seen = sightings(run, masks, points, p["visible"], p["contained"])
    node, agreed = consensus_merge(masks, node, label, seen, apart, p["consensus"], p["min_observers"], p["together"])
    keys = ("min_masks", "small", "far", "min_cover", "min_views", "min_arc")
    node, drops = drop(run, masks, boxes, node, **{k: p[k] for k in keys})
    label = {n: label[n] for n in np.unique(node[node >= 0]).tolist()}
    steps = {"splits": splits, "duplicates": duplicates, "merges": merges, "consensus_merges": agreed}
    return node, label, {**steps, "dropped": drops}
