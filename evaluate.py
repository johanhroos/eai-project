"""Score predicted object identities against the Replica ground truth.

ids gives one predicted object ID per mask: the SAM track (Masks.sam_id) or the graph node (Graph.node). A prediction's
segmentation in a frame is the union of its masks there, so a node that merged duplicates counts once. In a frame, a
prediction matches a GT object when their masks have IoU >= MATCH_IOU.

  IDF1      identity F1 (Ristani et al., 2016). Predictions and GT objects are paired one-to-one over the whole run so
            that the most frames match: IDTP. IDF1 = 2 IDTP / (predicted + GT detections), IDP = IDTP / predicted,
            IDR = IDTP / GT. Splitting an object over several IDs and merging objects into one ID both lower it.
  revisits  for every GT revisit (lift3d.compute_events): is the ID that matches the object last before it leaves the
            same as the one that matches it first after it comes back? kept or switched, or missed when no prediction
            matches it on one side within REVISIT_WINDOW frames.

GT objects of replica.IGNORE_CLASSES and GT detections under lift3d.MIN_PIXELS pixels are left out,
and so are the predictions that match them.

queries scores the object map as a text search with the OpenLex3D queries (see its docstring). It works per object:
OpenLex3D's own protocol scores point clouds instead.
"""

from collections import Counter, defaultdict

import numpy as np
import pycocotools.mask as mask_utils
from scipy.optimize import linear_sum_assignment

from graph import query
from utils import lift3d, replica
from utils.schema import Events, Masks, TextAligned

MATCH_IOU = 0.5
REVISIT_WINDOW = 100  # raw frames to look for a match before the object leaves and after it comes back


def evaluate(masks: Masks, ids: np.ndarray, gt: Masks, events: Events, objects: dict) -> dict:
    """Metrics of the predicted ids for masks against gt, the GT masks stage of the same run (see the docstring).

    objects is replica.load_objects(scene), events the events stage.
    """
    ignored = {o for o, obj in objects.items() if obj.class_name in replica.IGNORE_CLASSES}
    gt_area = lift3d.rle_area(gt.rle)
    matched = Counter()  # (prediction, GT object) -> frames they match in
    best = {}  # (frame, GT object) -> the matching prediction with the highest IoU
    n_pred = n_gt = 0
    for frame in np.union1d(masks.frame, gt.frame):
        p_rows, g_rows = np.flatnonzero(masks.frame == frame), np.flatnonzero(gt.frame == frame)
        preds = np.unique(ids[p_rows])
        objs = gt.sam_id[g_rows]
        counted = np.array([o not in ignored for o in objs.tolist()], bool) & (gt_area[g_rows] >= lift3d.MIN_PIXELS)
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
        "ids": len(np.unique(ids)),
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
            if (k1, k2) != (lift3d.KIND["hidden"], lift3d.KIND["revisited"]):
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
        p_rows, g_rows = np.flatnonzero(masks.frame == frame), np.flatnonzero(gt.frame == frame)
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
    object_vectors: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """How well the objects in ids are found by text: every query ranks all of them (graph.query).

    targets maps each query to the GT objects it describes (replica.load_queries) and text_emb holds the queries'
    embeddings in the same order. object_vectors optionally supplies saved node descriptors instead of reconstructing
    them from observation caches.
    Queries whose objects are never seen in the run (lift3d.MIN_PIXELS) are left out, and
    so are those objects. Walking down a query's ranking, an object is a hit when it stands for (owners) a target that
    is not found yet. Anything else counts against it: an object that stands for nothing, for another object, or for a
    target already found, which is a duplicate in the map.

      map        mean over queries of the average precision: the mean, over the query's targets, of the precision at
                 the rank where each is found, 0 for one never found
      top1       share of queries whose first object is a hit
      reachable  share of the targets that some object stands for: the recall a perfect ranking would get
    """
    seen = set(gt.sam_id[lift3d.rle_area(gt.rle) >= lift3d.MIN_PIXELS].tolist())
    keep = [i for i, objs in enumerate(targets.values()) if seen & set(objs)]
    sets = [seen & set(list(targets.values())[i]) for i in keep]
    owner = owners(masks, ids, gt)
    objects, object_emb = query.object_embeddings(masks, emb, ids) if object_vectors is None else object_vectors
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
COUNTS = {"ids", "kept", "switched", "missed"}  # printed as integers


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
