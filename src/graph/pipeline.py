"""Replay cached observations, persist compact objects, and rebuild on CPU."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils

from graph.association import AssociationConfig, ObjectMemory, ObjectNode, Observation, normalized, valid_box
from graph.node import fused_box
from utils.lift3d import Box3D
from utils.schema import Boxes, Features, Graph, Masks, PersistentObjects, Run, TextAligned, load_stage, save_stage

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "association.json"


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


@dataclass
class BuildResult:
    graph: Graph
    memory: ObjectMemory
    matches: list[dict]
    history: list[dict]


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
    matches, history = [], []
    frames = np.unique(masks.frame) if frames is None else np.unique(np.r_[frames, masks.frame])
    for frame in frames:
        if until is not None and frame > until:
            break
        rows = np.flatnonzero(masks.frame == frame)
        groups = observation_groups(masks, rows, memory.config.duplicate_iou)
        observations = []
        for group in groups:
            lo, hi = fused_box(boxes, group)
            points = int(boxes.n_points[group].max())
            label_votes = np.bincount(masks.prompt[group], weights=masks.score[group], minlength=len(vocab))
            observations.append(
                Observation(
                    mean_descriptor(group, features.dino),
                    Box3D(lo, hi, points),
                    vocab[int(label_votes.argmax())],
                    int(masks.sam_id[group[0]]),
                    mean_descriptor(group, siglip.siglip),
                )
            )
        ids = memory.associate_frame(int(frame), observations)
        for group, object_id, trace in zip(groups, ids, memory.last_matches, strict=True):
            assignments[group] = object_id
            matches.append({**trace, "rows": group.tolist(), "sam_ids": masks.sam_id[group].tolist()})
        history.append({"frame": int(frame), "nodes": len(memory.nodes), "observations": len(groups)})
    return BuildResult(Graph(assignments), memory, matches, history)


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

    dino, siglip = bank("descriptors"), bank("siglip_descriptors")
    sums = np.zeros((n, siglip.shape[2]), np.float32)
    lo, hi = np.full((n, 3), np.nan, np.float32), np.full((n, 3), np.nan, np.float32)
    counts = np.zeros(n, np.int64)
    for row, node in enumerate(nodes):
        if valid_box(node.box):
            lo[row], hi[row], counts[row] = node.box.min, node.box.max, node.box.n_points
        if node.siglip_sum is not None:
            sums[row] = node.siglip_sum
    return PersistentObjects(
        np.array([node.id for node in nodes], np.int32),
        lo,
        hi,
        counts,
        dino,
        siglip,
        sums,
        *[
            np.array([getattr(node, name) for node in nodes], np.int64)
            for name in ("siglip_count", "first_seen", "last_seen", "observation_count")
        ],
        [node.label for node in nodes],
        asdict(memory.config),
        memory.next_id,
        memory.last_frame,
    )


def save_memory(path: Path, memory: ObjectMemory) -> None:
    save_stage(path, snapshot(memory), memory.config.metadata())


def load_memory(path: Path) -> ObjectMemory:
    data = load_stage(path, PersistentObjects)
    memory = ObjectMemory(AssociationConfig(**data.config))
    memory.next_id, memory.last_frame = data.next_id, data.last_frame
    memory.dino_dimension = data.dino.shape[2] or None
    memory.siglip_dimension = data.siglip.shape[2] or None
    for row, object_id in enumerate(data.object_id):
        node = ObjectNode(int(object_id), data.labels[row])
        if data.n_points[row] > 0:
            node.box = Box3D(data.box_min[row].copy(), data.box_max[row].copy(), int(data.n_points[row]))
        node.descriptors = [e.copy() for e in data.dino[row] if normalized(e) is not None]
        node.siglip_descriptors = [e.copy() for e in data.siglip[row] if normalized(e) is not None]
        node.siglip_sum = data.siglip_sum[row].copy() if data.siglip_count[row] else None
        for name in ("siglip_count", "first_seen", "last_seen", "observation_count"):
            setattr(node, name, int(getattr(data, name)[row]))
        memory.nodes[node.id] = node
    return memory


def save_result(path: Path, result: BuildResult) -> None:
    if np.any(result.graph.node < 0):
        raise ValueError("cannot save an incomplete prefix as the full graph")
    path.mkdir(parents=True, exist_ok=True)
    # Remove completion marker before writing any of the related artifacts.
    (path / "metadata.json").unlink(missing_ok=True)
    save_memory(path / "memory", result.memory)
    for name, data in (("matches", result.matches), ("history", result.history), ("edges", result.memory.edges())):
        (path / f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    save_stage(path, result.graph, result.memory.config.metadata())


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

    result = replay(*cached_observations(run), config=config, frames=footage.frame_numbers(run))
    if save:
        save_result(run.stage_dir("association"), result)
    return result


def main():
    import argparse

    from utils import footage, vocab

    parser = argparse.ArgumentParser(description="Rebuild graph from cached observations without loading any models")
    parser.add_argument("--source", default="replica/room_0/00")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--mask-source", default="sam3", choices=["sam3", "gt"])
    parser.add_argument("--pose-source", default="gt", choices=["gt", "vggt"])
    parser.add_argument("--vocab")
    parser.add_argument("--association-config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run = footage.make_run(args.source, args.stride, args.mask_source, args.pose_source, vocab.tag(args.vocab))
    result = rebuild(run, AssociationConfig.from_json(args.association_config))
    print(
        f"Saved {len(result.memory.nodes)} objects and {len(result.memory.edges())} edges to {run.stage_dir('association')}"
    )


if __name__ == "__main__":
    main()
