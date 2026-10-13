"""Incremental static-object association in a shared metric world frame: SAM's tracks, checked by DINO and location.

SAM 3's tracker is mostly right about which masks are one object, so a mask whose SAM ID already belongs to a node
joins that node. DINO appearance and 3D location decide the rest:
  cut    where a track's box is more than cut_gap from the box of its last reliable mask, or it looks different from
         that mask (DINO cosine under cut_dino) and the two boxes don't touch (cut_dino_gap), or it matches another
         node switch_margin better than its own, SAM's ID went over to another object: the rest of the track is matched
         like a new one. SAM switches between objects that look alike (identical chairs) and between neighbours (a
         cushion and its sofa), so each cue finds switches the others miss. A DINO drop with the boxes still touching
         is mostly the same object seen from another side: cutting there too found 73% of SAM's switches on Replica
         instead of 69%, but also cut 3.2% of the steps within one object instead of 0.7%. A wrong cut costs little:
         the merge below joins the pieces again. In a frame where several tracks move over cut_gap at once
         (bad_frame_tracks), the camera pose is off, and nothing is cut by these cues. With cut_after above 1, a track
         is cut only where it jumps by the gap or DINO in that many reliable masks in a row, and until then a mask that
         jumped stays with the node as a member only, like an unreliable mask. That keeps a one-frame jump from making
         a node of one mask (on IMG_0649, 20 such nodes instead of 34 with 2), but such a jump is mostly SAM's mask on
         another object for a frame (211 of 606 switches on Replica go back the next mask), which then lands in the
         wrong node: on Replica, IDF1 fell from 0.598 to 0.588 and GT objects per node rose from 1.15 to 1.28. A node
         takes one mask per frame: of two masks whose tracks lead to it, the one that matches it less is cut.
  match  a new track, or the rest of a cut one, joins the node in memory it matches, or becomes a new node
  merge  a node that gets a reliable view joins an older node it matches, unless the two were ever seen in one frame:
         an object back in view under a new SAM ID, or one first seen too far away to match

A match scores DINO appearance and location. The DINO cosine is clipped to [0, 1] and geometry is in [0, 1]: how
much of the smaller box lies inside the other (lift3d.box_containment), and how close the centres are. Containment
rather than IoU, since a view of part of an object lies inside the object's box, which has grown to all its views, and
IoU would score it near 0. Enabled nonnegative weights are normalized once. Missing cues score zero with no per-pair
weight renormalization, so a match with less evidence cannot inflate its score.

Only reliable masks (lift3d.reliable: enough depth, and confident where the depth has a confidence) are evidence.
The others join their track's node but don't add to its box or DINO descriptors, and don't cut, match or merge: a
far view's depth smears along the camera ray and its crop is a few pixels. A node's box is around the pooled points of
its reliable masks without the outer 2% (graph.nodes.node_box), so it covers every part any view saw, but no single
view sets it. The saved graph draws each node's box again from every view of it (graph/multiview.py), which needs all
the frames. SigLIP takes no part in matching; each node sums its observations' SigLIP embeddings for text search.
"""

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from geometry.lift3d import MIN_CONFIDENCE, Box3D, box_containment
from graph.nodes import fit_box


def box_gap(a: Box3D, b: Box3D) -> float:
    """Metres between two boxes, 0 when they touch or overlap."""
    return float(np.linalg.norm(np.clip(np.maximum(a.min - b.max, b.min - a.max), 0, None)))


@dataclass(frozen=True)
class AssociationConfig:
    dino_weight: float = 0.45
    box_weight: float = 0.4
    enable_dino: bool = True
    enable_geometry: bool = True
    threshold: float = 0.55  # above each weight alone (0.53, 0.47): DINO and location must both agree
    max_descriptors: int = 4
    max_candidate_distance: float | None = 1.0  # metres between box centres; None disables gate
    geometry_inside_weight: float = 0.5  # 1 = containment only; 0 = centre distance only
    center_distance_scale: float = 0.5  # metres; similarity exp(-distance / scale)
    descriptor_duplicate_similarity: float = 0.95
    require_same_label: bool = False
    duplicate_iou: float = 0.6
    follow_tracks: bool = True  # a mask joins its SAM track's node; False matches every mask on its own, never merges
    cut_gap: float = 0.3  # metres between a track's box and its last reliable one's at which the track is cut
    cut_dino: float = 0.5  # DINO cosine to the track's last reliable mask under which it is cut
    cut_dino_gap: float | None = 0.0  # metres the boxes must also be apart for a DINO cut; None: DINO alone cuts
    cut_after: int = 1  # reliable masks in a row a track must jump in (by cut_gap or cut_dino) before it is cut
    bad_frame_tracks: int = 3  # a frame where this many tracks, and at least half, move over cut_gap cuts none
    switch_margin: float | None = 0.2  # score by which another node must match a track's mask better to cut it

    def __post_init__(self):
        weights = (self.dino_weight, self.box_weight)
        if not all(np.isfinite(w) and w >= 0 for w in weights):
            raise ValueError("weights must be finite and nonnegative")
        if sum(self.effective_weights) <= 0:
            raise ValueError("at least one enabled component must have positive weight")
        for value in (
            self.threshold,
            self.geometry_inside_weight,
            self.descriptor_duplicate_similarity,
            self.duplicate_iou,
            self.cut_dino,
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("thresholds and fractions must be between zero and one")
        for count in (self.max_descriptors, self.bad_frame_tracks, self.cut_after):
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError("max_descriptors, bad_frame_tracks and cut_after must be positive integers")
        for scale in (self.center_distance_scale, self.cut_gap):
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("center_distance_scale and cut_gap must be finite and positive")
        if self.max_candidate_distance is not None and (
            not np.isfinite(self.max_candidate_distance) or self.max_candidate_distance <= 0
        ):
            raise ValueError("max_candidate_distance must be positive or null")
        if self.switch_margin is not None and (not np.isfinite(self.switch_margin) or self.switch_margin < 0):
            raise ValueError("switch_margin must be nonnegative or null")
        if self.cut_dino_gap is not None and (not np.isfinite(self.cut_dino_gap) or self.cut_dino_gap < 0):
            raise ValueError("cut_dino_gap must be nonnegative or null")
        for enabled in (self.enable_dino, self.enable_geometry, self.require_same_label, self.follow_tracks):
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
        # The reliable masks depend on lift3d.MIN_CONFIDENCE too, so a new value rebuilds the graph
        return {"algorithm": "track-association-v2", "config": asdict(self), "min_confidence": MIN_CONFIDENCE}


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
    """One object seen in one frame: the masks of one duplicate group (graph.build.observation_groups)."""

    dino: np.ndarray | None
    box: Box3D | None
    label: str | None = None
    sam_ids: tuple[int, ...] | int | None = ()  # the SAM tracks of its masks: which node it follows, never an ID
    siglip: np.ndarray | None = None
    reliable: bool = True  # whether its depth is trusted (lift3d.reliable); if not, it only follows its track
    points: np.ndarray | None = None  # (k, 3) its reliable masks' points; None: the corners of its box
    conf: float = float("nan")  # confidence of its depth, which picks a node's box while no view is reliable

    def __post_init__(self):
        if self.box is not None:
            try:
                self.box = Box3D(
                    np.asarray(self.box.min, np.float32), np.asarray(self.box.max, np.float32), int(self.box.n_points)
                )
            except (TypeError, ValueError, OverflowError):
                self.box = None
        if self.sam_ids is None:
            self.sam_ids = ()
        elif np.ndim(self.sam_ids) == 0:
            self.sam_ids = (int(self.sam_ids),)
        else:
            self.sam_ids = tuple(int(s) for s in self.sam_ids)
        self.reliable = bool(self.reliable) and valid_box(self.box)
        if self.reliable and self.points is None:
            self.points = np.stack([self.box.min, self.box.max])
        if self.points is not None:
            points = np.asarray(self.points, np.float32).reshape(-1, 3)
            self.points = points[np.isfinite(points).all(axis=1)]
            self.reliable &= len(self.points) > 0


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
    points: np.ndarray | None = None  # (k, 3) float32, its reliable observations' points, which its box is around
    fallback: Box3D | None = None  # the most confident box among its observations, its box while none is reliable
    fallback_conf: float = -np.inf
    frames: set[int] = field(default_factory=set)  # the frames it was seen in: nodes seen in one are two objects

    @property
    def siglip(self):
        return normalized(self.siglip_sum)

    @property
    def reliable(self) -> bool:
        return self.points is not None and len(self.points) > 0

    def refresh_box(self) -> None:
        if self.reliable:
            self.box = Box3D(*fit_box(self.points), len(self.points))
        elif self.fallback is not None:
            self.box = Box3D(self.fallback.min.copy(), self.fallback.max.copy(), self.fallback.n_points)


class ObjectMemory:
    """Bounded per-object attributes, monotonic IDs, no deletion on disappearance.

    Merges leave gaps in the IDs; renumber() closes them. Greedy strongest-pair-first matching is deterministic and
    one-to-one within a frame. Match traces contain selected/best candidate scores before updates.
    """

    def __init__(self, config: AssociationConfig | None = None):
        self.config = config or AssociationConfig()
        self.nodes: dict[int, ObjectNode] = {}
        self.next_id = 0
        self.last_frame: int | None = None
        self.last_matches: list[dict] = []
        self.last_merges: list[dict] = []
        self.weights = self.config.effective_weights
        self.dino_dimension: int | None = None
        self.siglip_dimension: int | None = None
        self.tracks: dict[int, int] = {}  # SAM ID -> the node its masks join
        self.track_last: dict[int, tuple[np.ndarray | None, Box3D]] = {}  # SAM ID -> DINO, box of last reliable mask
        self.merged: dict[int, int] = {}  # node ID -> the node it was merged into
        self.jumps: dict[int, int] = {}  # SAM ID -> reliable masks in a row it jumped in, while under cut_after

    def find(self, object_id: int) -> int:
        """The node an ID stands for now, after merges."""
        while object_id in self.merged:
            object_id = self.merged[object_id]
        return object_id

    def components(self, observation: Observation, node: ObjectNode) -> dict:
        return self._pair([observation.dino], observation.box, observation.label, node)

    def _pair(self, descriptors, box: Box3D | None, label: str | None, node: ObjectNode) -> dict:
        """The match scores of an observation's, or another node's, descriptors and box against a node."""
        c = self.config
        compatible = not (c.require_same_label and label and node.label and label != node.label)
        empty = {
            "dino": 0.0,
            "inside": 0.0,
            "center": 0.0,
            "distance": None,
            "geometry": 0.0,
            "score": 0.0,
            "compatible": False,
        }
        if not compatible:
            return empty
        distance = None
        inside = center = 0.0
        if valid_box(box) and valid_box(node.box):
            distance = float(np.linalg.norm(box.center - node.box.center))
            if c.max_candidate_distance is not None and distance > c.max_candidate_distance:
                return {**empty, "distance": distance}
            inside = float(box_containment(box, node.box))
            inside = float(np.clip(inside, 0, 1)) if np.isfinite(inside) else 0.0
            center = float(np.exp(-distance / c.center_distance_scale))
        dino = 0.0
        if c.enable_dino:
            dino = max((prototype_similarity(d, node.descriptors) for d in descriptors), default=0.0)
        w = c.geometry_inside_weight
        geometry = w * inside + (1 - w) * center if c.enable_geometry else 0.0
        wd, wg = self.weights
        score = float(wd * dino + wg * geometry)
        return {
            "dino": dino,
            "inside": inside,
            "center": center,
            "distance": distance,
            "geometry": geometry,
            "score": score,
            "compatible": bool(compatible),
        }

    def score(self, observation: Observation, node: ObjectNode) -> float:
        result = self.components(observation, node)
        return result["score"] if result["compatible"] else -np.inf

    def _move(self, observation: Observation) -> tuple[float, float] | None:
        """(DINO cosine, metres between the boxes) from a reliable observation to its tracks' last reliable masks.

        The most similar and nearest of them. None when it isn't reliable or its tracks have no reliable mask yet.
        """
        lasts = [self.track_last[s] for s in observation.sam_ids if s in self.track_last]
        if not observation.reliable or not lasts:
            return None
        e = normalized(observation.dino)
        cosines = [float(e @ d) for d, _ in lasts if e is not None and d is not None and e.shape == d.shape]
        gap = min(box_gap(observation.box, box) for _, box in lasts)
        return max(cosines, default=1.0), gap  # without DINO, appearance can't tell a switch

    def _jumped(self, dino: float, gap: float) -> bool:
        """Whether a track's move (_move) cuts it.

        It is with a gap over cut_gap, or a DINO cosine under cut_dino where the boxes are also over cut_dino_gap apart.
        """
        c = self.config
        return gap > c.cut_gap or (dino < c.cut_dino and (c.cut_dino_gap is None or gap > c.cut_dino_gap))

    def _rival(self, observation: Observation, object_id: int) -> dict | None:
        """The other node a track's mask matches switch_margin better than its own, as {"own", "rival", "score"}.

        None if there is none. SAM switches between neighbours that look alike, where neither the gap nor DINO tell.
        """
        c = self.config
        if c.switch_margin is None:
            return None
        own = self.components(observation, self.nodes[object_id])["score"]  # 0 when over max_candidate_distance away
        best = None
        for other_id, other in self.nodes.items():
            if other_id != object_id and other.reliable:
                score = self.score(observation, other)
                if score > c.threshold and score > own + c.switch_margin and (best is None or score > best[1]):
                    best = (other_id, score)
        return None if best is None else {"own": own, "rival": best[0], "score": best[1]}

    def associate_frame(self, frame: int, observations: list[Observation]) -> np.ndarray:
        if self.last_frame is not None and frame <= self.last_frame:
            raise ValueError("frames must be strictly increasing; submit all observations together")
        c = self.config
        self.last_merges = []
        # SAM's tracks first: a mask whose track has a node joins it, unless the track jumped to another object
        follow = {}
        if c.follow_tracks:
            for row, observation in enumerate(observations):
                nodes = sorted({self.find(self.tracks[s]) for s in observation.sam_ids if s in self.tracks})
                if nodes:
                    follow[row] = nodes
        moves = {row: self._move(observations[row]) for row in follow}
        moved = [m for m in moves.values() if m is not None]
        jumped = sum(gap > c.cut_gap for _, gap in moved)
        bad_pose = jumped >= c.bad_frame_tracks and 2 * jumped >= len(moved)
        followed, cuts, held = {}, {}, {}
        observations = list(observations)
        for row, nodes in follow.items():
            move = moves[row]
            sam_ids = observations[row].sam_ids
            if not bad_pose and move is not None:
                if not self._jumped(*move):
                    for s in sam_ids:
                        self.jumps.pop(s, None)
                else:
                    jumps = 1 + max((self.jumps.get(s, 0) for s in sam_ids), default=0)
                    if jumps >= c.cut_after:
                        for s in sam_ids:
                            self.jumps.pop(s, None)
                        cuts[row] = {"dino": move[0], "gap": move[1]}
                        continue
                    # Not cut yet (cut_after): it stays with the node, but only as its member, like an unreliable mask
                    for s in sam_ids:
                        self.jumps[s] = jumps
                    held[row] = {"dino": move[0], "gap": move[1], "jumps": jumps}
                    observations[row] = replace(observations[row], reliable=False)
            object_id = self.find(nodes[0])
            reliable = not bad_pose and move is not None and row not in held
            switch = self._rival(observations[row], object_id) if reliable else None
            if switch is not None:
                cuts[row] = {"dino": move[0], "gap": move[1], **switch}
                continue
            # A mask on two nodes' tracks follows the older node, and the other track comes along. Merging the two
            # nodes there put three monitors up to 2.7 m apart into one node on IMG_0649; on Replica it changed nothing.
            followed[row] = object_id
        # One mask per node in a frame: two masks whose tracks lead to one node (one track was cut and its rest matched
        # there, or the two shared a mask) are two objects. The best match keeps the node, a reliable mask before an
        # unreliable one; the others are cut.
        by_node = {}
        for row, object_id in followed.items():
            by_node.setdefault(object_id, []).append(row)
        for object_id, rows in by_node.items():
            if len(rows) > 1:
                scores = {row: self.components(observations[row], self.nodes[object_id])["score"] for row in rows}
                keep = max(rows, key=lambda row: (observations[row].reliable, scores[row], -row))
                for row in rows:
                    if row != keep:
                        del followed[row]
                        cuts[row] = {"taken": object_id, "score": scores[row], "kept_score": scores[keep]}

        # The rest, new tracks and cut ones, match among the reliable nodes not seen in this frame yet
        claimed = {self.find(object_id) for object_id in followed.values()}
        pairs, best = [], {}
        for row, observation in enumerate(observations):
            if row in followed or not observation.reliable:
                continue
            for object_id, node in self.nodes.items():
                if object_id in claimed or not node.reliable:
                    continue
                parts = self.components(observation, node)
                if not parts["compatible"]:
                    continue
                if row not in best or parts["score"] > best[row][1]["score"]:
                    best[row] = (object_id, parts)
                if parts["score"] > c.threshold:
                    pairs.append((-parts["score"], row, object_id, parts))
        assigned, used = {}, set()
        for _, row, object_id, parts in sorted(pairs, key=lambda p: p[:3]):
            if row not in assigned and object_id not in used:
                assigned[row] = (object_id, parts)
                used.add(object_id)

        ids = np.empty(len(observations), np.int32)
        self.last_matches = []
        evidence = []  # nodes with a reliable view in this frame, which may merge into an older node
        for row, observation in enumerate(observations):
            if row in followed:
                node, via = self.nodes[self.find(followed[row])], "track"
                candidate_id, parts = node.id, None
            elif row in assigned:
                (candidate_id, parts), via = assigned[row], "match"
                node = self.nodes[candidate_id]
            else:
                candidate_id, parts = best.get(row, (None, None))
                node, via = ObjectNode(self.next_id, observation.label, first_seen=frame), "new"
                self.nodes[node.id] = node
                self.next_id += 1
            self.last_matches.append(
                {
                    "frame": int(frame),
                    "object_id": node.id,
                    "candidate_id": candidate_id,
                    "matched": via != "new",
                    "via": via,
                    "cut": cuts.get(row),
                    "held": held.get(row),
                    "reliable": observation.reliable,
                    "scores": parts,
                }
            )
            self._update(node, observation, frame, track=not bad_pose)
            ids[row] = node.id
            if observation.reliable:
                evidence.append(node.id)
        if c.follow_tracks:
            for object_id in dict.fromkeys(evidence):
                self._merge_into_older(self.find(object_id), frame)
        self.last_frame = int(frame)
        return np.array([self.find(i) for i in ids], np.int32)

    def _merge_into_older(self, object_id: int, frame: int) -> None:
        """Merge a node into the reliable node it matches best, if they were never seen in one frame."""
        node = self.nodes[object_id]
        best = None
        for other_id, other in self.nodes.items():
            if other_id == object_id or not other.reliable:
                continue
            parts = self._pair(node.descriptors, node.box, node.label, other)
            if not parts["compatible"] or parts["score"] <= self.config.threshold:
                continue
            if node.frames.isdisjoint(other.frames) and (best is None or parts["score"] > best[1]["score"]):
                best = (other_id, parts)
        if best is not None:
            self._merge(object_id, best[0], frame, "match", best[1])

    def _merge(self, a: int, b: int, frame: int, reason: str, scores: dict | None = None) -> int | None:
        """Merge two nodes into the older one and return its ID, or None if they were seen in one frame."""
        a, b = self.find(a), self.find(b)
        if a == b:
            return a
        keep, gone = self.nodes[min(a, b)], self.nodes[max(a, b)]
        if not keep.frames.isdisjoint(gone.frames):
            return None
        if gone.reliable:
            keep.points = gone.points if keep.points is None else np.concatenate([keep.points, gone.points])
        if gone.fallback_conf > keep.fallback_conf:
            keep.fallback, keep.fallback_conf = gone.fallback, gone.fallback_conf
        for descriptor in gone.descriptors:
            keep.descriptors = update_bank(keep.descriptors, descriptor, self.config)
        if gone.siglip_sum is not None:
            if keep.siglip_sum is None:
                keep.siglip_sum = gone.siglip_sum.copy()
            elif keep.siglip_sum.shape == gone.siglip_sum.shape:
                keep.siglip_sum = keep.siglip_sum + gone.siglip_sum
            keep.siglip_count += gone.siglip_count
        keep.first_seen, keep.last_seen = min(keep.first_seen, gone.first_seen), max(keep.last_seen, gone.last_seen)
        keep.observation_count += gone.observation_count
        keep.frames |= gone.frames
        keep.label = keep.label or gone.label
        keep.refresh_box()
        self.merged[gone.id] = keep.id
        del self.nodes[gone.id]
        for s, object_id in self.tracks.items():
            if object_id == gone.id:
                self.tracks[s] = keep.id
        self.last_merges.append(
            {"frame": int(frame), "kept": keep.id, "merged": gone.id, "reason": reason, "scores": scores}
        )
        return keep.id

    def renumber(self) -> dict[int, int]:
        """Number the nodes 0, 1, ... in order of first appearance again, closing the gaps merges left.

        Returns the new ID of every ID given so far, a merged one's being that of the node it went into.
        """
        new = {old: k for k, old in enumerate(sorted(self.nodes))}
        mapping = {old: new[self.find(old)] for old in range(self.next_id)}
        self.nodes = {new[old]: node for old, node in self.nodes.items()}
        for node in self.nodes.values():
            node.id = mapping[node.id]
        self.tracks = {s: mapping[object_id] for s, object_id in self.tracks.items()}
        self.merged = {}
        self.next_id = len(self.nodes)
        return mapping

    def _update(self, node: ObjectNode, observation: Observation, frame: int, track: bool = True) -> None:
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
        if e is not None and (node.siglip_sum is None or e.shape == node.siglip_sum.shape):
            node.siglip_sum = e.copy() if node.siglip_sum is None else node.siglip_sum + e
            node.siglip_count += 1
        if observation.reliable:
            node.descriptors = update_bank(node.descriptors, dino, c)
            points = observation.points
            node.points = points.copy() if node.points is None else np.concatenate([node.points, points])
        elif valid_box(observation.box):
            conf = observation.conf if np.isfinite(observation.conf) else -np.inf
            if node.fallback is None or conf > node.fallback_conf:
                node.fallback, node.fallback_conf = observation.box, conf
        node.refresh_box()
        for s in observation.sam_ids:
            self.tracks[s] = node.id
            if observation.reliable and track:
                self.track_last[s] = (dino, observation.box)
        node.label = node.label or observation.label
        node.last_seen = int(frame)
        node.observation_count += 1
        node.frames.add(int(frame))
