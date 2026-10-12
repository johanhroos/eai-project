"""Incremental static-object association in a shared metric world frame.

Whether an observation is an object already in memory is decided by its DINO appearance and its location only: SAM
IDs and labels don't decide it. The DINO cosine is clipped to [0, 1] and geometry is in [0, 1]. Enabled nonnegative
weights are normalized once. Missing cues score zero with no per-pair weight renormalization, so a match with less
evidence cannot inflate its score.

A node's box is the union of every box it was matched with: the smallest minimum and the largest maximum per axis, so
it grows to the whole space the object fills as views of its parts come in, also behind occluders, and never shrinks.
SigLIP takes no part in matching; each node sums its observations' SigLIP embeddings for text search.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from utils.lift3d import Box3D, box_iou_3d


@dataclass(frozen=True)
class AssociationConfig:
    dino_weight: float = 0.45
    box_weight: float = 0.4
    enable_dino: bool = True
    enable_geometry: bool = True
    threshold: float = 0.55  # above each weight alone (0.53, 0.47): DINO and location must both agree
    max_descriptors: int = 4
    max_candidate_distance: float | None = 1.0  # metres between box centres; None disables gate
    geometry_iou_weight: float = 0.5  # 1 = IoU only; 0 = centre distance only
    center_distance_scale: float = 0.5  # metres; similarity exp(-distance / scale)
    descriptor_duplicate_similarity: float = 0.95
    require_same_label: bool = False
    duplicate_iou: float = 0.6

    def __post_init__(self):
        weights = (self.dino_weight, self.box_weight)
        if not all(np.isfinite(w) and w >= 0 for w in weights):
            raise ValueError("weights must be finite and nonnegative")
        if sum(self.effective_weights) <= 0:
            raise ValueError("at least one enabled component must have positive weight")
        for value in (
            self.threshold,
            self.geometry_iou_weight,
            self.descriptor_duplicate_similarity,
            self.duplicate_iou,
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("thresholds and fractions must be between zero and one")
        if (
            not isinstance(self.max_descriptors, int)
            or isinstance(self.max_descriptors, bool)
            or self.max_descriptors < 1
        ):
            raise ValueError("max_descriptors must be a positive integer")
        if not np.isfinite(self.center_distance_scale) or self.center_distance_scale <= 0:
            raise ValueError("center_distance_scale must be finite and positive")
        if self.max_candidate_distance is not None and (
            not np.isfinite(self.max_candidate_distance) or self.max_candidate_distance <= 0
        ):
            raise ValueError("max_candidate_distance must be positive or null")
        for enabled in (self.enable_dino, self.enable_geometry, self.require_same_label):
            if not isinstance(enabled, bool):
                raise ValueError("component switches must be booleans")

    @property
    def effective_weights(self) -> tuple[float, float]:
        raw = np.array([self.dino_weight if self.enable_dino else 0, self.box_weight if self.enable_geometry else 0])
        total = raw.sum()
        return tuple((raw / total if total > 0 else raw).tolist())

    @classmethod
    def from_json(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text()))

    def metadata(self) -> dict:
        return {"algorithm": "persistent-association-v3", "config": asdict(self)}


def normalized(descriptor: np.ndarray | None) -> np.ndarray | None:
    if descriptor is None:
        return None
    try:
        e = np.asarray(descriptor, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if e.ndim != 1 or not e.size or not np.isfinite(e).all():
        return None
    norm = float(np.linalg.norm(e))
    return e / norm if np.isfinite(norm) and norm > 1e-6 else None


def valid_box(box: Box3D | None) -> bool:
    return (
        box is not None
        and np.shape(box.min) == (3,)
        and np.shape(box.max) == (3,)
        and np.isfinite(box.min).all()
        and np.isfinite(box.max).all()
        and np.all(box.max >= box.min)
        and box.n_points > 0
    )


def prototype_similarity(descriptor, bank) -> float:
    e = normalized(descriptor)
    if e is None:
        return 0.0
    sims = []
    for descriptor in bank:
        stored = normalized(descriptor)
        if stored is not None and e.shape == stored.shape:
            sims.append(float(np.clip(e @ stored, 0, 1)))
    return max(sims, default=0.0)


def update_bank(bank, descriptor, config):
    e = normalized(descriptor)
    if e is None or (bank and e.shape != bank[0].shape):
        return bank
    if bank and prototype_similarity(e, bank) >= config.descriptor_duplicate_similarity:
        return bank
    candidates = [*bank, e.astype(np.float16)]
    if len(candidates) > config.max_descriptors:
        stack = np.stack(candidates).astype(np.float32)
        matrix = stack @ stack.T
        np.fill_diagonal(matrix, -np.inf)
        candidates.pop(int(matrix.max(axis=1).argmax()))
    return candidates


@dataclass
class Observation:
    dino: np.ndarray | None
    box: Box3D | None
    label: str | None = None
    sam_id: int | None = None  # metadata only; never an object ID
    siglip: np.ndarray | None = None

    def __post_init__(self):
        if self.box is not None:
            try:
                self.box = Box3D(
                    np.asarray(self.box.min, np.float32), np.asarray(self.box.max, np.float32), int(self.box.n_points)
                )
            except (TypeError, ValueError, OverflowError):
                self.box = None


@dataclass
class ObjectNode:
    id: int
    label: str | None
    box: Box3D | None = None
    descriptors: list[np.ndarray] = field(default_factory=list)
    first_seen: int = 0
    last_seen: int = 0
    observation_count: int = 0
    siglip_sum: np.ndarray | None = None
    siglip_count: int = 0

    @property
    def siglip(self):
        return normalized(self.siglip_sum)


class ObjectMemory:
    """Bounded per-object attributes, monotonic IDs, no deletion on disappearance.

    Greedy strongest-pair-first assignment is deterministic and one-to-one within
    a frame. Match traces contain selected/best candidate scores before updates.
    """

    def __init__(self, config: AssociationConfig | None = None):
        self.config = config or AssociationConfig()
        self.nodes: dict[int, ObjectNode] = {}
        self.next_id = 0
        self.last_frame: int | None = None
        self.last_matches: list[dict] = []
        self.weights = self.config.effective_weights
        self.dino_dimension: int | None = None
        self.siglip_dimension: int | None = None

    def components(self, observation: Observation, node: ObjectNode) -> dict:
        c = self.config
        compatible = not (c.require_same_label and observation.label and node.label and observation.label != node.label)
        empty = {
            "dino": 0.0,
            "iou": 0.0,
            "center": 0.0,
            "distance": None,
            "geometry": 0.0,
            "score": 0.0,
            "compatible": False,
        }
        if not compatible:
            return empty
        distance = None
        iou = center = 0.0
        if valid_box(observation.box) and valid_box(node.box):
            distance = float(np.linalg.norm(observation.box.center - node.box.center))
            if c.max_candidate_distance is not None and distance > c.max_candidate_distance:
                return {**empty, "distance": distance}
            iou = float(box_iou_3d(observation.box, node.box))
            iou = float(np.clip(iou, 0, 1)) if np.isfinite(iou) else 0.0
            center = float(np.exp(-distance / c.center_distance_scale))
        dino = prototype_similarity(observation.dino, node.descriptors) if c.enable_dino else 0.0
        geometry = (c.geometry_iou_weight * iou + (1 - c.geometry_iou_weight) * center) if c.enable_geometry else 0.0
        wd, wg = self.weights
        score = float(wd * dino + wg * geometry)
        return {
            "dino": dino,
            "iou": iou,
            "center": center,
            "distance": distance,
            "geometry": geometry,
            "score": score,
            "compatible": bool(compatible),
        }

    def score(self, observation: Observation, node: ObjectNode) -> float:
        result = self.components(observation, node)
        return result["score"] if result["compatible"] else -np.inf

    def associate_frame(self, frame: int, observations: list[Observation]) -> np.ndarray:
        if self.last_frame is not None and frame <= self.last_frame:
            raise ValueError("frames must be strictly increasing; submit all observations together")
        pairs, best = [], {}
        for row, observation in enumerate(observations):
            for object_id, node in self.nodes.items():
                parts = self.components(observation, node)
                if not parts["compatible"]:
                    continue
                if row not in best or parts["score"] > best[row][1]["score"]:
                    best[row] = (object_id, parts)
                if parts["score"] > self.config.threshold:
                    pairs.append((-parts["score"], row, object_id, parts))
        assigned, used = {}, set()
        for _, row, object_id, parts in sorted(pairs, key=lambda p: p[:3]):
            if row not in assigned and object_id not in used:
                assigned[row] = (object_id, parts)
                used.add(object_id)
        ids = np.empty(len(observations), np.int32)
        self.last_matches = []
        for row, observation in enumerate(observations):
            matched = row in assigned
            candidate_id, parts = assigned.get(row, best.get(row, (None, None)))
            if matched:
                node = self.nodes[candidate_id]
            else:
                node = ObjectNode(self.next_id, observation.label, first_seen=frame)
                self.nodes[node.id] = node
                self.next_id += 1
            self.last_matches.append(
                {
                    "frame": int(frame),
                    "object_id": node.id,
                    "candidate_id": candidate_id,
                    "matched": matched,
                    "scores": parts,
                }
            )
            self._update(node, observation, frame)
            ids[row] = node.id
        self.last_frame = int(frame)
        return ids

    def _update(self, node: ObjectNode, observation: Observation, frame: int) -> None:
        c = self.config
        dino = normalized(observation.dino)
        e = normalized(observation.siglip)
        if dino is not None:
            self.dino_dimension = self.dino_dimension or len(dino)
            if len(dino) != self.dino_dimension:
                dino = None
        if e is not None:
            self.siglip_dimension = self.siglip_dimension or len(e)
            if len(e) != self.siglip_dimension:
                e = None
        node.descriptors = update_bank(node.descriptors, dino, c)
        if e is not None and (node.siglip_sum is None or e.shape == node.siglip_sum.shape):
            node.siglip_sum = e.copy() if node.siglip_sum is None else node.siglip_sum + e
            node.siglip_count += 1
        if valid_box(observation.box):
            incoming = observation.box
            if not valid_box(node.box):
                node.box = Box3D(incoming.min.copy(), incoming.max.copy(), incoming.n_points)
            else:  # the union: the object fills every part any view saw
                lo, hi = np.minimum(node.box.min, incoming.min), np.maximum(node.box.max, incoming.max)
                node.box = Box3D(lo, hi, max(node.box.n_points, incoming.n_points))
        node.label = node.label or observation.label
        node.last_seen = int(frame)
        node.observation_count += 1
