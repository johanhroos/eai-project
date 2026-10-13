"""Replay cached observations, persist compact objects, and rebuild on CPU.

The one model a rebuild may load is SigLIP 2's text encoder, on the GPU, once per vocabulary (vocab_text).
"""

import copy
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils

from graph import refine, shape
from graph.association import AssociationConfig, ObjectMemory, ObjectNode, Observation, normalized, valid_box
from graph.node import fit_box, mask_points, most_confident
from utils.lift3d import Box3D, reliable
from utils.schema import Boxes, Features, Graph, Masks, PersistentObjects, Run, TextAligned, load_stage, save_stage

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "association.json"
SNAPSHOT_POINTS = 512  # of each node's pooled points kept in a saved memory, so a restored node's box can grow


def mean_descriptor(rows, values):
    valid = [normalized(values[r]) for r in rows]
    valid = [e for e in valid if e is not None]
    return normalized(np.sum(valid, axis=0)) if valid else None


def observation_groups(masks: Masks, rows: np.ndarray, threshold: float) -> list[np.ndarray]:
    """Complete-link duplicate groups avoid transitive bridges between objects.

    Only consolidate strongly overlapping masks in the same frame. Different
    prompts may describe the same object; labels don't gate duplicate detection.
    """
    if not len(rows):
        return []
    rles = [masks.rle[r] for r in rows]
    iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rows)))
    groups = []
    for i in np.argsort(-masks.score[rows], kind="stable"):
        for group in groups:
            if np.all(iou[i, group] >= threshold):
                group.append(int(i))
                break
        else:
            groups.append([int(i)])
    return [rows[group] for group in groups]


def observation(masks: Masks, boxes: Boxes, features: Features, siglip: TextAligned, vocab, trusted, group):
    """The Observation of a duplicate group's masks (rows). trusted is lift3d.reliable(boxes).

    Its box is around its reliable masks' points, else that of its mask with the most confident depth.
    """
    good = group[trusted[group]]
    points = mask_points(boxes, good) if len(good) else None
    best = most_confident(boxes, group)
    if points is not None:
        box = Box3D(*fit_box(points), len(points))
    elif best is not None:
        box = Box3D(boxes.box_min[best], boxes.box_max[best], int(boxes.n_points[best]))
    else:
        box = None
    conf = float(boxes.conf[best]) if best is not None and boxes.conf is not None else float("nan")
    label_votes = np.bincount(masks.prompt[group], weights=masks.score[group], minlength=len(vocab))
    return Observation(
        mean_descriptor(good if len(good) else group, features.dino),
        box,
        vocab[int(label_votes.argmax())],
        tuple(masks.sam_id[group].tolist()),
        mean_descriptor(group, siglip.siglip),
        reliable=points is not None,
        points=points,
        conf=conf,
    )


@dataclass
class BuildResult:
    graph: Graph
    memory: ObjectMemory
    matches: list[dict]
    history: list[dict]
    merges: list[dict]


def replay(
    masks: Masks,
    boxes: Boxes,
    features: Features,
    siglip: TextAligned,
    vocab: list[str],
    config: AssociationConfig | None = None,
    frames=None,
    until: int | None = None,
) -> BuildResult:
    n = len(masks.frame)
    if any(len(a) != n for a in (boxes.n_points, features.dino, siglip.siglip)):
        raise ValueError("all observation stages must be row-aligned")
    if n and (masks.prompt.min() < 0 or masks.prompt.max() >= len(vocab)):
        raise ValueError("mask prompt index outside vocabulary")
    memory = ObjectMemory(config)
    assignments = np.full(n, -1, np.int32)
    matches, history, merges = [], [], []
    trusted = reliable(boxes)
    frames = np.unique(masks.frame) if frames is None else np.unique(np.r_[frames, masks.frame])
    for frame in frames:
        if until is not None and frame > until:
            break
        rows = np.flatnonzero(masks.frame == frame)
        groups = observation_groups(masks, rows, memory.config.duplicate_iou)
        observations = [observation(masks, boxes, features, siglip, vocab, trusted, group) for group in groups]
        ids = memory.associate_frame(int(frame), observations)
        for group, object_id, trace in zip(groups, ids, memory.last_matches, strict=True):
            assignments[group] = object_id
            matches.append({**trace, "rows": group.tolist(), "sam_ids": masks.sam_id[group].tolist()})
        merges += memory.last_merges
        history.append({"frame": int(frame), "nodes": len(memory.nodes), "observations": len(groups)})
    # Masks of a node that was merged go to the node it went into. IDs stay those of the memory, in order of creation
    # with gaps where nodes merged, so a prefix replay (until) shows its nodes under the IDs of the full one.
    done = assignments >= 0
    assignments[done] = [memory.find(int(i)) for i in assignments[done]]
    for trace in matches:
        for key in ("object_id", "candidate_id"):
            trace[key] = None if trace[key] is None else memory.find(trace[key])
    return BuildResult(Graph(assignments), memory, matches, history, merges)


def snapshot(memory: ObjectMemory) -> PersistentObjects:
    nodes = sorted(memory.nodes.values(), key=lambda n: n.id)
    n, k = len(nodes), memory.config.max_descriptors

    def bank(name):
        dimension = max((len(e) for node in nodes for e in getattr(node, name)), default=0)
        data = np.zeros((n, k, dimension), np.float16)
        for row, node in enumerate(nodes):
            for i, e in enumerate(getattr(node, name)):
                data[row, i] = e
        return data

    dino = bank("descriptors")
    sums = np.zeros((n, memory.siglip_dimension or 0), np.float32)
    lo, hi = np.full((n, 3), np.nan, np.float32), np.full((n, 3), np.nan, np.float32)
    counts = np.zeros(n, np.int64)
    points = np.full((n, SNAPSHOT_POINTS, 3), np.nan, np.float16)
    for row, node in enumerate(nodes):
        if valid_box(node.box):
            lo[row], hi[row], counts[row] = node.box.min, node.box.max, node.box.n_points
        if node.siglip_sum is not None:
            sums[row] = node.siglip_sum
        if node.reliable:
            keep = np.unique(np.linspace(0, len(node.points) - 1, SNAPSHOT_POINTS).astype(int))
            points[row, : len(keep)] = node.points[keep]
    return PersistentObjects(
        np.array([node.id for node in nodes], np.int32),
        lo,
        hi,
        counts,
        dino,
        sums,
        *[
            np.array([getattr(node, name) for node in nodes], np.int64)
            for name in ("siglip_count", "first_seen", "last_seen", "observation_count")
        ],
        [node.label for node in nodes],
        asdict(memory.config),
        memory.next_id,
        memory.last_frame,
        points,
        np.array(list(memory.tracks), np.int32),
        np.array([memory.find(i) for i in memory.tracks.values()], np.int32),
        [sorted(node.frames) for node in nodes],
    )


def save_memory(path: Path, memory: ObjectMemory) -> None:
    save_stage(path, snapshot(memory), memory.config.metadata())


def load_memory(path: Path) -> ObjectMemory:
    data = load_stage(path, PersistentObjects)
    memory = ObjectMemory(AssociationConfig(**data.config))
    memory.next_id, memory.last_frame = data.next_id, data.last_frame
    memory.dino_dimension = data.dino.shape[2] or None
    memory.siglip_dimension = data.siglip_sum.shape[1] or None
    for row, object_id in enumerate(data.object_id):
        node = ObjectNode(int(object_id), data.labels[row])
        if data.n_points[row] > 0:
            node.box = Box3D(data.box_min[row].copy(), data.box_max[row].copy(), int(data.n_points[row]))
            node.fallback = node.box
        if data.points is not None:
            pts = data.points[row].astype(np.float32)
            node.points = pts[np.isfinite(pts).all(axis=1)] if np.isfinite(pts).any() else None
        node.frames = set(data.frames[row]) if data.frames is not None else set()
        node.descriptors = [e.copy() for e in data.dino[row] if normalized(e) is not None]
        node.siglip_sum = data.siglip_sum[row].copy() if data.siglip_count[row] else None
        for name in ("siglip_count", "first_seen", "last_seen", "observation_count"):
            setattr(node, name, int(getattr(data, name)[row]))
        memory.nodes[node.id] = node
    if data.track_id is not None:
        memory.tracks = dict(zip(data.track_id.tolist(), data.track_node.tolist(), strict=True))
    return memory


def metadata(config: AssociationConfig, finished: bool = True) -> dict:
    """The graph stage's metadata: the memory's settings, and those of finish's steps if they ran."""
    return {
        **config.metadata(),
        "refine": refine.PARAMS if finished else None,
        "object_boxes": shape.PARAMS if finished else None,
    }


Finish = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]]


def vocab_text(run: Run, vocab: list[str]) -> np.ndarray | None:
    """(V, D) SigLIP text embeddings of the masks stage's prompts, cached next to the siglip stage.

    The first call for a vocabulary loads SigLIP 2's text encoder; later ones read the cache. None when nothing is
    cached and there is no GPU to encode them: the labels are then SAM's votes alone.
    """
    import torch

    from models import siglip

    path = run.stage_dir("siglip") / "vocab_text.npz"
    if path.exists():
        with np.load(path) as cached:
            if cached["vocab"].tolist() == vocab and str(cached["model"]) == siglip.MODEL_ID:
                return cached["text"]
    if not torch.cuda.is_available():
        return None
    text = siglip.load_encoder().encode_text(vocab).cpu().numpy()
    np.savez(path, vocab=np.array(vocab), model=np.array(siglip.MODEL_ID), text=text)
    return text


def finish(
    run: Run, masks: Masks, boxes: Boxes, features: Features, emb: TextAligned, vocab: list[str], node: np.ndarray
):
    """The graph stage's last steps, on the memory's node of every mask, once every frame is in.

    graph/refine.py splits, merges and drops nodes (-1 for the masks of dropped ones) and labels them from all their
    views (SAM's votes weighed by SigLIP), the nodes left are numbered 0, 1, ..., and graph/shape.py draws each one's
    box from every view of it. Returns the node of every mask; box_min, box_max and the label (an index into vocab) by
    node ID; and refine's record of what it did, with "graph_ids" from its node IDs to the saved ones.
    """
    text = vocab_text(run, vocab)
    refined, label, record = refine.refine(run, masks, boxes, features, emb, text, node)
    record["labels"] = "SAM votes" if text is None else "SAM votes weighed by SigLIP"
    used = np.unique(refined[refined >= 0])
    final = np.where(refined >= 0, np.searchsorted(used, refined), -1).astype(np.int32)
    record["graph_ids"] = {int(n): k for k, n in enumerate(used.tolist())}
    labels = np.array([label[n] for n in used.tolist()], np.int16)
    return (final, *shape.object_boxes(run, masks, boxes, final), labels, record)


def save_result(path: Path, result: BuildResult, finish: Finish | None = None) -> None:
    """Save the graph stage with its node IDs renumbered 0, 1, ... (graph.query and graph.edges index nodes by ID).

    finish(node), such as pipeline.finish for the run, takes the memory's node of every mask and gives the saved one
    (-1 for none), the saved nodes' boxes and labels, and a record, saved as refine.json. Without it the memory's nodes
    are saved as they are, without boxes or labels: graph.node.build_nodes then draws the boxes around the pooled
    points and counts SAM's votes for the labels. memory/ holds the
    memory as the video left it, by its own IDs; refine.json's "memory_to_graph" maps each to the saved node that holds
    most of its masks (null if it holds none). The traces get the saved IDs: a match's object_id is the node of its
    masks, and a merge's trace keeps the IDs of the time and adds the saved node as "node".
    """
    if np.any(result.graph.node < 0):
        raise ValueError("cannot save an incomplete prefix as the full graph")
    path.mkdir(parents=True, exist_ok=True)
    # Remove completion marker before writing any of the related artifacts.
    (path / "metadata.json").unlink(missing_ok=True)
    memory = copy.deepcopy(result.memory)
    mapping = memory.renumber()
    in_memory = np.array([mapping[int(i)] for i in result.graph.node], np.int32)
    node, lo, hi, label, record = finish(in_memory) if finish is not None else (in_memory, None, None, None, None)
    saved = {}  # the memory's IDs -> the saved node holding most of its masks
    for m in np.unique(in_memory).tolist():
        kept = node[(in_memory == m) & (node >= 0)]
        saved[m] = int(np.bincount(kept).argmax()) if len(kept) else None
    matches = []
    for m in result.matches:
        kept = node[m["rows"]][node[m["rows"]] >= 0]
        object_id = int(np.bincount(kept).argmax()) if len(kept) else None
        candidate = None if m["candidate_id"] is None else saved.get(mapping[m["candidate_id"]])
        matches.append({**m, "object_id": object_id, "candidate_id": candidate})
    merges = [{**m, "node": saved.get(mapping[m["kept"]])} for m in result.merges]
    save_memory(path / "memory", memory)
    (path / "refine.json").unlink(missing_ok=True)
    traces = [("matches", matches), ("history", result.history), ("merges", merges)]
    if record is not None:
        traces.append(("refine", {**record, "memory_to_graph": saved}))
    for name, data in traces:
        (path / f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    save_stage(path, Graph(node, lo, hi, label), metadata(result.memory.config, finish is not None))


def cached_observations(run: Run):
    required = (("masks", Masks), ("boxes", Boxes), ("features", Features), ("siglip", TextAligned))
    missing = [stage for stage, _ in required if not run.done(stage)]
    if missing:
        raise FileNotFoundError(f"Missing cached stages: {', '.join(missing)}. Run run.py on this source first.")
    mask_time = (run.stage_dir("masks") / "metadata.json").stat().st_mtime
    older = [stage for stage, _ in required if (run.stage_dir(stage) / "metadata.json").stat().st_mtime < mask_time]
    if older:
        raise ValueError(f"Cached stages older than masks: {', '.join(older)}. Refresh them with run.py first.")
    data = [load_stage(run.stage_dir(stage), cls) for stage, cls in required]
    vocab = json.loads((run.stage_dir("masks") / "metadata.json").read_text())["vocab"]
    return (*data, vocab)


def rebuild(run: Run, config: AssociationConfig | None = None, save: bool = True) -> BuildResult:
    # Source frame numbers include frames with no detections.
    from utils import footage

    observations = cached_observations(run)
    result = replay(*observations, config=config, frames=footage.frame_numbers(run))
    if save:
        save_result(run.stage_dir("graph"), result, partial(finish, run, *observations))
    return result


def main():
    import argparse

    from utils import footage, vocab

    parser = argparse.ArgumentParser(description="Rebuild graph from cached observations (see the module docstring)")
    parser.add_argument("--source", default="replica/room_0/00")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--mask-source", default="sam3", choices=["sam3", "gt"])
    parser.add_argument("--pose-source", default="gt", choices=["gt", "vggt"])
    parser.add_argument("--vocab")
    parser.add_argument("--association-config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run = footage.make_run(args.source, args.stride, args.mask_source, args.pose_source, vocab.tag(args.vocab))
    rebuild(run, AssociationConfig.from_json(args.association_config))
    node = load_stage(run.stage_dir("graph"), Graph).node
    print(f"Saved {len(np.unique(node[node >= 0]))} objects to {run.stage_dir('graph')}")


if __name__ == "__main__":
    main()
