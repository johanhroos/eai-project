"""Score predicted object identities against the Replica ground truth.

ids gives one predicted object ID per mask: the SAM track (Masks.sam_id) or the graph node (Graph.node), -1 for a mask
the graph left out (graph/refine.py), which predicts nothing. A prediction's
segmentation in a frame is the union of its masks there, so a node that merged duplicates counts once. In a frame, a
prediction matches a GT object when their masks have IoU >= MATCH_IOU.

  IDF1      identity F1 (Ristani et al., 2016). Predictions and GT objects are paired one-to-one over the whole run so
            that the most frames match: IDTP. IDF1 = 2 IDTP / (predicted + GT detections), IDP = IDTP / predicted,
            IDR = IDTP / GT. Splitting an object over several IDs and merging objects into one ID both lower it.
  revisits  for every GT revisit (ground_truth.compute_events): is the ID that matches the object last before it
            leaves the same as the one that matches it first after it comes back? kept or switched, or missed when no
            prediction matches it on one side within REVISIT_WINDOW frames.

GT objects of replica.IGNORE_CLASSES and GT detections under ground_truth.MIN_PIXELS pixels are left out,
and so are the predictions that match them.

queries scores the object map as a text search with the OpenLex3D queries (see its docstring). It works per object:
OpenLex3D's own protocol scores point clouds instead. nearest_objects scores the nearest-object query against the GT
boxes.
"""

from collections import Counter, defaultdict

import numpy as np
import pycocotools.mask as mask_utils
from scipy.optimize import linear_sum_assignment

from data import replica
from data.stages import Boxes, Events, Graph, Masks, ObjectBoxes, TextAligned, rle_area
from evaluation import ground_truth
from graph import edges, query
from graph.nodes import build_nodes

MATCH_IOU = 0.5
REVISIT_WINDOW = 100  # raw frames to look for a match before the object leaves and after it comes back


def evaluate(masks: Masks, ids: np.ndarray, gt: Masks, events: Events, objects: dict) -> dict:
    """Metrics of the predicted ids for masks against gt, the GT masks stage of the same run (see the docstring).

    objects is replica.load_objects(scene), events the events stage.
    """
    ignored = {o for o, obj in objects.items() if obj.class_name in replica.IGNORE_CLASSES}
    gt_area = rle_area(gt.rle)
    matched = Counter()  # (prediction, GT object) -> frames they match in
    best = {}  # (frame, GT object) -> the matching prediction with the highest IoU
    n_pred = n_gt = 0
    for frame in np.union1d(masks.frame, gt.frame):
        p_rows, g_rows = np.flatnonzero((masks.frame == frame) & (ids >= 0)), np.flatnonzero(gt.frame == frame)
        preds = np.unique(ids[p_rows])
        objs = gt.sam_id[g_rows]
        counted = np.array([o not in ignored for o in objs.tolist()], bool) & (
            gt_area[g_rows] >= ground_truth.MIN_PIXELS
        )
        n_gt += int(counted.sum())
        if not len(preds):
            continue
        pred_rle = [mask_utils.merge([masks.rle[r] for r in p_rows[ids[p_rows] == p]]) for p in preds]
        iou = np.zeros((len(preds), len(g_rows)))
        if len(g_rows):
            iou = np.asarray(mask_utils.iou(pred_rle, [gt.rle[r] for r in g_rows], [0] * len(g_rows)))
        hit = iou >= MATCH_IOU
        n_pred += int((~(hit & ~counted).any(axis=1)).sum())  # predictions of ignored objects don't count
        for pi, gi in zip(*np.nonzero(hit & counted), strict=True):
            matched[int(preds[pi]), int(objs[gi])] += 1
        for gi in np.flatnonzero(counted & hit.any(axis=0)):
            best[int(frame), int(objs[gi])] = int(preds[iou[:, gi].argmax()])

    pred_ids = sorted({p for p, _ in matched})
    obj_ids = sorted({o for _, o in matched})
    counts = np.zeros((len(pred_ids), len(obj_ids)))
    for (p, o), n in matched.items():
        counts[pred_ids.index(p), obj_ids.index(o)] = n
    rows, cols = linear_sum_assignment(counts, maximize=True)
    idtp = float(counts[rows, cols].sum())

    return {
        "idf1": 2 * idtp / max(n_pred + n_gt, 1),
        "idp": idtp / max(n_pred, 1),
        "idr": idtp / max(n_gt, 1),
        "ids": len(np.unique(ids[ids >= 0])),
        "ids_per_object": float((counts > 0).sum(0).mean()) if obj_ids else 0.0,
        "objects_per_id": float((counts > 0).sum(1).mean()) if pred_ids else 0.0,
        **revisits(best, events, ignored),
    }


def revisits(best: dict, events: Events, ignored: set) -> dict:
    """Count the GT revisits whose prediction is kept, switched or missed. best maps (frame, object) to a prediction."""
    frames_of = defaultdict(list)
    for frame, o in sorted(best):
        frames_of[o].append(frame)
    events_of = defaultdict(list)  # events are in frame order, so each object's alternate hidden, revisited
    for o, frame, kind in zip(events.object_id.tolist(), events.frame.tolist(), events.kind.tolist(), strict=True):
        if o not in ignored:
            events_of[o].append((frame, kind))

    outcome = Counter(kept=0, switched=0, missed=0)
    for o, ev in events_of.items():
        for (left, k1), (back, k2) in zip(ev[::2], ev[1::2], strict=True):
            if (k1, k2) != (ground_truth.KIND["hidden"], ground_truth.KIND["revisited"]):
                raise ValueError(f"object {o}: events out of order at frames {left}, {back}")
            before = [f for f in frames_of[o] if left - REVISIT_WINDOW < f <= left]
            after = [f for f in frames_of[o] if back <= f < back + REVISIT_WINDOW]
            if not before or not after:
                outcome["missed"] += 1
            else:
                outcome["kept" if best[before[-1], o] == best[after[0], o] else "switched"] += 1
    n = outcome["kept"] + outcome["switched"]
    return {"revisits": sum(outcome.values()), **outcome, "kept_rate": outcome["kept"] / n if n else float("nan")}


def owners(masks: Masks, ids: np.ndarray, gt: Masks) -> dict[int, int]:
    """The GT object each predicted ID stands for: the one its masks match (IoU >= MATCH_IOU) in the most frames.

    No class is ignored here, since queries can ask for walls or ceilings. IDs that never match have no owner.
    """
    votes = defaultdict(Counter)
    for frame in np.intersect1d(masks.frame, gt.frame):
        p_rows, g_rows = np.flatnonzero((masks.frame == frame) & (ids >= 0)), np.flatnonzero(gt.frame == frame)
        if not len(p_rows):
            continue
        preds = np.unique(ids[p_rows])
        pred_rle = [mask_utils.merge([masks.rle[r] for r in p_rows[ids[p_rows] == p]]) for p in preds]
        iou = np.asarray(mask_utils.iou(pred_rle, [gt.rle[r] for r in g_rows], [0] * len(g_rows)))
        for pi in np.flatnonzero(iou.max(axis=1) >= MATCH_IOU):
            votes[int(preds[pi])][int(gt.sam_id[g_rows[iou[pi].argmax()]])] += 1
    return {p: c.most_common(1)[0][0] for p, c in votes.items()}


def queries(
    masks: Masks,
    ids: np.ndarray,
    gt: Masks,
    emb: TextAligned,
    targets: dict[str, list[int]],
    text_emb: np.ndarray,
) -> dict:
    """How well the objects in ids are found by text: every query ranks all of them (graph.query).

    targets maps each query to the GT objects it describes (replica.load_queries) and text_emb holds the queries'
    embeddings in the same order. Queries whose objects are never seen in the run (ground_truth.MIN_PIXELS) are left
    out, and so are those objects. Walking down a query's ranking, an object is a hit when it stands for (owners) a
    target that is not found yet. Anything else counts against it: an object that stands for nothing, for another
    object, or for a target already found, which is a duplicate in the map.

      map        mean over queries of the average precision: the mean, over the query's targets, of the precision at
                 the rank where each is found, 0 for one never found
      top1       share of queries whose first object is a hit
      reachable  share of the targets that some object stands for: the recall a perfect ranking would get
    """
    seen = set(gt.sam_id[rle_area(gt.rle) >= ground_truth.MIN_PIXELS].tolist())
    keep = [i for i, objs in enumerate(targets.values()) if seen & set(objs)]
    sets = [seen & set(list(targets.values())[i]) for i in keep]
    owner = owners(masks, ids, gt)
    objects, object_emb = query.object_embeddings(masks, emb, ids)
    order, _ = query.rank(text_emb[keep], object_emb)
    stands_for = np.array([owner.get(int(o), -1) for o in objects])

    aps, top1 = [], []
    for ranked, tset in zip(stands_for[order], sets, strict=True):
        found, ap = set(), 0.0
        for k, o in enumerate(ranked.tolist(), start=1):
            if o in tset and o not in found:
                found.add(o)
                ap += len(found) / k
        aps.append(ap / len(tset))
        top1.append(bool(len(ranked)) and ranked[0] in tset)
    owned = {owner[int(o)] for o in objects if int(o) in owner}
    return {
        "map": float(np.mean(aps)),
        "top1": float(np.mean(top1)),
        "reachable": float(np.mean([len(t & owned) / len(t) for t in sets])),
        "queries": len(sets),
    }


def nearest_objects(
    masks: Masks,
    ids: np.ndarray | Graph,
    gt: Masks,
    boxes: Boxes,
    vocab: list[str],
    gt_boxes: ObjectBoxes,
    objects: dict,
) -> dict:
    """How often the map finds the nearest object of a type (graph.query --nearest), against the GT boxes.

    Every GT object A that is seen in the run (ground_truth.MIN_PIXELS) asks for the nearest object of every class C
    that has at least two other seen objects, so that it matters which one comes first. The answer is the C object
    nearest to A by their GT boxes (gt_boxes stage), ranked as graph.edges.nearest ranks nodes. In the map, A is the
    node that stands for it (owners) and is seen in the most frames, and the candidates are ranked by their boxes
    (graph.nodes). The query is right when the first candidate stands for the answer. A query whose A no node stands for
    is wrong. Nodes that stand for A itself are never candidates: a duplicate of the anchor isn't another object.

      labels  the candidates are the nodes labelled C, as for graph.query --nearest
      oracle  the candidates are the nodes that stand for an object of class C: only the map's boxes and identities
      anchor  share of the queries whose A some node stands for
      reach   share of the queries whose A and answer some node stands for: what oracle gets with perfect boxes

    Classes in replica.IGNORE_CLASSES are left out. The labels column needs SAM prompts that are Replica class names,
    as on Replica without --vocab. ids is the node ID of every mask, or the graph stage, whose nodes then get its
    boxes and labels.
    """
    graph = ids if isinstance(ids, Graph) else Graph(node=np.asarray(ids, np.int32))
    used = np.unique(graph.node[graph.node >= 0])
    ids = np.where(graph.node >= 0, np.searchsorted(used, graph.node), -1)  # 0, 1, ..., so IDs index nodes
    own = (graph.box_min[used], graph.box_max[used]) if graph.box_min is not None else (None, None)
    label = graph.label[used] if graph.label is not None else None
    nodes = build_nodes(masks, boxes, Graph(ids.astype(np.int32), *own, label), vocab)
    owner = owners(masks, ids, gt)
    stands_for = np.array([owner.get(n.id, -1) for n in nodes])
    node_class = np.array([objects[o].class_name if o >= 0 else "" for o in stands_for.tolist()])
    labels = np.array([n.label for n in nodes])
    n_frames = np.array([len(n.frames) for n in nodes])

    seen = set(gt.sam_id[rle_area(gt.rle) >= ground_truth.MIN_PIXELS].tolist())
    keep = [
        k
        for k, o in enumerate(gt_boxes.object_id.tolist())
        if o in seen and objects[o].class_name not in replica.IGNORE_CLASSES
    ]
    gt_ids, lo, hi = gt_boxes.object_id[keep], gt_boxes.box_min[keep], gt_boxes.box_max[keep]
    gt_class = np.array([objects[int(o)].class_name for o in gt_ids])

    hits, anchored, reached = {"labels": [], "oracle": []}, [], []
    for a, obj in enumerate(gt_ids.tolist()):
        for c in np.unique(gt_class):
            others = [k for k in np.flatnonzero(gt_class == c).tolist() if k != a]
            if len(others) < 2:
                continue
            answer = int(gt_ids[edges.nearest_boxes(lo, hi, a, others)[0][0]])
            owning = np.flatnonzero(stands_for == obj)
            anchored.append(len(owning) > 0)
            reached.append(len(owning) > 0 and answer in stands_for)
            if not len(owning):
                hits["labels"].append(False)
                hits["oracle"].append(False)
                continue
            anchor = int(owning[n_frames[owning].argmax()])
            for name, typed in (("labels", labels == c), ("oracle", node_class == c)):
                ranked = edges.nearest(nodes, anchor, np.flatnonzero(typed & (stands_for != obj)).tolist())
                hits[name].append(bool(ranked) and int(stands_for[ranked[0][0]]) == answer)
    mean = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
    return {
        "near_labels": mean(hits["labels"]),
        "near_oracle": mean(hits["oracle"]),
        "near_anchor": mean(anchored),
        "near_reach": mean(reached),
        "near_queries": len(anchored),
    }


IDENTITY_COLUMNS = {
    "idf1": "IDF1",
    "idp": "IDP",
    "idr": "IDR",
    "ids": "IDs",
    "ids_per_object": "IDs/obj",
    "objects_per_id": "obj/ID",
    "kept_rate": "kept%",
    "kept": "kept",
    "switched": "switch",
    "missed": "missed",
}
QUERY_COLUMNS = {
    "level0_map": "L0 mAP",
    "level0_top1": "L0 top1",
    "level1_map": "L1 mAP",
    "level1_top1": "L1 top1",
    "level0_reachable": "reach",
}
NEAR_COLUMNS = {
    "near_labels": "labels",
    "near_oracle": "oracle",
    "near_anchor": "anchor",
    "near_reach": "reach",
    "near_queries": "queries",
}
COUNTS = {"ids", "kept", "switched", "missed", "near_queries"}  # printed as integers


def table(results: dict[str, dict[str, dict]], columns: dict[str, str] = IDENTITY_COLUMNS) -> str:
    """Results by run and then by method as a text table, with the mean over runs when there are several.

    columns maps the metrics to show to their headings.
    """
    width = max(len(m) for r in results.values() for m in r) + 2
    lines = [f"{'':<{width}}" + "".join(f"{h:>9}" for h in columns.values())]

    def row(name, m):
        cells = [f"{m[k]:>9.0f}" if k in COUNTS else f"{m[k]:>9.2f}" for k in columns]
        return f"{name:<{width}}" + "".join(cells)

    for source, methods in results.items():
        lines.append(source)
        lines += [row(f"  {name}", m) for name, m in methods.items()]
    if len(results) > 1:
        lines.append(f"mean over {len(results)} runs")
        for name in next(iter(results.values())):
            lines.append(row(f"  {name}", {k: np.nanmean([r[name][k] for r in results.values()]) for k in columns}))
    return "\n".join(lines)
