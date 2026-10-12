"""Identity diagnostics for cached Replica runs (no model inference)."""

import json
from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np
import pycocotools.mask as mask_utils

from utils import replica
from utils.lift3d import rle_area
from utils.schema import Events, Graph, Masks, load_stage


def audit(masks, ids, gt, objects, min_frames=3, events=None):
    """Find persistent splits/mixed identities using GT mask IoU >= 0.5.

    These are diagnostic counts, not proof of every physical merge. Ignore the
    same background classes as identity evaluation. Require repeated matches to
    exclude one-frame segmentation noise from the example lists.
    """
    ignored = {o for o, obj in objects.items() if obj.class_name in replica.IGNORE_CLASSES}
    counts = Counter()
    best = {}
    areas = rle_area(gt.rle)
    for frame in np.intersect1d(masks.frame, gt.frame):
        rows = np.flatnonzero(masks.frame == frame)
        gt_rows = np.flatnonzero((gt.frame == frame) & (areas >= 500))
        gt_rows = np.array([row for row in gt_rows if gt.sam_id[row] not in ignored], int)
        if not len(gt_rows):
            continue
        nodes = np.unique(ids[rows])
        unions = [mask_utils.merge([masks.rle[r] for r in rows[ids[rows] == node]]) for node in nodes]
        overlap = np.asarray(mask_utils.iou(unions, [gt.rle[r] for r in gt_rows], [0] * len(gt_rows)))
        for i, j in zip(*np.nonzero(overlap >= 0.5), strict=True):
            counts[int(nodes[i]), int(gt.sam_id[gt_rows[j]])] += 1
        for j in np.flatnonzero((overlap >= 0.5).any(axis=0)):
            i = int(overlap[:, j].argmax())
            best[int(frame), int(gt.sam_id[gt_rows[j]])] = int(nodes[i])
    by_node, by_gt = defaultdict(list), defaultdict(list)
    for (node, object_id), frames in counts.items():
        if frames >= min_frames:
            by_node[node].append({"gt_id": object_id, "frames": frames})
            by_gt[object_id].append({"node": node, "frames": frames})
    mixed = {node: entries for node, entries in by_node.items() if len(entries) > 1}
    split = {object_id: entries for object_id, entries in by_gt.items() if len(entries) > 1}
    examples = []
    if events is not None:
        event_pairs = defaultdict(list)
        for object_id, frame, kind in zip(events.object_id, events.frame, events.kind, strict=True):
            if int(object_id) not in ignored:
                event_pairs[int(object_id)].append((int(frame), int(kind)))
        for object_id, sequence in event_pairs.items():
            for (left, _), (back, _) in zip(sequence[::2], sequence[1::2], strict=True):
                before = [f for f, o in best if o == object_id and left - 100 < f <= left]
                after = [f for f, o in best if o == object_id and back <= f < back + 100]
                if not before or not after:
                    continue
                f1, f2 = max(before), min(after)
                n1, n2 = best[f1, object_id], best[f2, object_id]
                examples.append(
                    {
                        "gt_id": object_id,
                        "label": objects[object_id].class_name,
                        "before_frame": f1,
                        "after_frame": f2,
                        "before_node": n1,
                        "after_node": n2,
                        "retained": n1 == n2,
                        "before_sam_ids": np.unique(masks.sam_id[(masks.frame == f1) & (ids == n1)]).tolist(),
                        "after_sam_ids": np.unique(masks.sam_id[(masks.frame == f2) & (ids == n2)]).tolist(),
                    }
                )
    return {
        "minimum_matching_frames": min_frames,
        "mixed_nodes": mixed,
        "split_gt_objects": split,
        "mixed_node_count": len(mixed),
        "split_gt_object_count": len(split),
        "revisit_examples": examples,
    }


def validate_run(run, save=True):
    # evaluate.py is a repository entry point rather than an installed module.
    import evaluate

    masks = load_stage(run.stage_dir("masks"), Masks)
    graph = load_stage(run.stage_dir("association"), Graph)
    gt = load_stage(replace(run, mask_source="gt", vocab="").stage_dir("masks"), Masks)
    events = load_stage(run.stage_dir("events"), Events)
    objects = replica.load_objects(replica.parse_source(run.source)[0])
    results = {}
    for method, ids in (("tracks", masks.sam_id), ("persistent_graph", graph.node)):
        results[method] = {
            "metrics": evaluate.evaluate(masks, ids, gt, events, objects),
            "audit": audit(masks, ids, gt, objects, events=events),
        }
    if save:
        path = run.stage_dir("eval")
        path.mkdir(parents=True, exist_ok=True)
        (path / "identity_validation.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    import argparse

    from utils.schema import Run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="replica/room_0/00")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--mask-source", default="sam3", choices=["sam3", "gt"])
    parser.add_argument("--vocab", default="")
    args = parser.parse_args()
    results = validate_run(Run(args.source, args.stride, args.mask_source, "gt", args.vocab))
    for method, values in results.items():
        print(method, values["metrics"])
        print(
            "Repeated GT matches: mixed nodes",
            values["audit"]["mixed_node_count"],
            "split GT objects",
            values["audit"]["split_gt_object_count"],
        )


if __name__ == "__main__":
    main()
