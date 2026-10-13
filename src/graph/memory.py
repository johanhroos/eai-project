"""Incremental static-object association in a shared metric world frame: SAM's tracks, checked by DINO and location.

SAM 3's tracker is mostly right about which masks are one object, so a mask whose SAM ID already belongs to a node
joins that node. DINO appearance and 3D location decide the rest:
  cut    where a track's box is more than CUT_GAP from the box of its last reliable mask, or it looks different from
         that mask (DINO cosine under CUT_DINO) and the two boxes don't touch (CUT_DINO_GAP), or it matches another
         node SWITCH_MARGIN better than its own, SAM's ID went over to another object: the rest of the track is matched
         like a new one. SAM switches between objects that look alike (identical chairs) and between neighbours (a
         cushion and its sofa), so each cue finds switches the others miss. A wrong cut costs little: the merge below
         joins the pieces again. In a frame where several tracks move over CUT_GAP at once (BAD_FRAME_TRACKS), the
         camera pose is off, and nothing is cut by these cues. With CUT_AFTER above 1, a track is cut only where it
         jumps by the gap or DINO in that many reliable masks in a row, and until then a mask that jumped stays with
         the node as a member only, like an unreliable mask. A node takes one mask per frame: of two masks whose tracks
         lead to it, the one that matches it less is cut. Why the constants have the values they have is measured in
         the README (the graph stage, step 3).
  match  a new track, or the rest of a cut one, joins the node in memory it matches, or becomes a new node
  merge  a node that gets a reliable view joins an older node it matches, unless the two were ever seen in one frame:
         an object back in view under a new SAM ID, or one first seen too far away to match

A match scores DINO appearance and location. The DINO cosine is clipped to [0, 1] and geometry is in [0, 1]: how
much of the smaller box lies inside the other (mask_boxes.box_containment), and how close the centres are. Containment
rather than IoU, since a view of part of an object lies inside the object's box, which has grown to all its views, and
IoU would score it near 0. The two weights are normalized to sum to 1. Missing cues score zero with no per-pair
weight renormalization, so a match with less evidence cannot inflate its score.

Only reliable masks (mask_boxes.reliable: enough depth, and confident where the depth has a confidence) are evidence.
The others join their track's node but don't add to its box or DINO descriptors, and don't cut, match or merge: a
far view's depth smears along the camera ray and its crop is a few pixels. A node's box is around the pooled points of
its reliable masks without the outer 2% (graph.nodes.node_box), so it covers every part any view saw, but no single
view sets it. The saved graph draws each node's box again from every view of it (graph/multiview.py), which needs all
the frames. SigLIP takes no part in matching; each node sums its observations' SigLIP embeddings for text search.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from geometry.mask_boxes import MIN_CONFIDENCE, Box3D, box_containment
from graph.nodes import fit_box


def box_gap(a: Box3D, b: Box3D) -> float:
    """Metres between two boxes, 0 when they touch or overlap."""
    return float(np.linalg.norm(np.clip(np.maximum(a.min - b.max, b.min - a.max), 0, None)))


DINO_WEIGHT = 0.45  # of a match's score: DINO appearance; the two weights are normalized to sum to 1 (WEIGHTS)
BOX_WEIGHT = 0.4  # of a match's score: location
THRESHOLD = 0.55  # above each weight alone (0.53, 0.47): DINO and location must both agree
MAX_DESCRIPTORS = 4  # DINO descriptors a node keeps: the most different of its reliable views'
MAX_CANDIDATE_DISTANCE = 1.0  # metres between box centres beyond which a node is no candidate
GEOMETRY_INSIDE_WEIGHT = 0.5  # of location: 1 = containment only; 0 = centre distance only
CENTER_DISTANCE_SCALE = 0.5  # metres; centre similarity exp(-distance / scale)
DESCRIPTOR_DUPLICATE_SIMILARITY = 0.95  # DINO cosine to a kept descriptor above which a new one isn't kept
DUPLICATE_IOU = 0.6  # masks of one frame that overlap this much are one observation (graph.build.observation_groups)
CUT_GAP = 0.3  # metres between a track's box and its last reliable one's at which the track is cut
CUT_DINO = 0.5  # DINO cosine to the track's last reliable mask under which it is cut
CUT_DINO_GAP = 0.0  # metres the boxes must also be apart for a DINO cut
CUT_AFTER = 1  # reliable masks in a row a track must jump in (by CUT_GAP or CUT_DINO) before it is cut
BAD_FRAME_TRACKS = 3  # a frame where this many tracks, and at least half, move over CUT_GAP cuts none
SWITCH_MARGIN = 0.2  # score by which another node must match a track's mask better to cut it
WEIGHTS = tuple((np.array([DINO_WEIGHT, BOX_WEIGHT]) / np.sum([DINO_WEIGHT, BOX_WEIGHT])).tolist())
PARAMS = {
    "dino_weight": DINO_WEIGHT,
    "box_weight": BOX_WEIGHT,
    "threshold": THRESHOLD,
    "max_descriptors": MAX_DESCRIPTORS,
    "max_candidate_distance": MAX_CANDIDATE_DISTANCE,
    "geometry_inside_weight": GEOMETRY_INSIDE_WEIGHT,
    "center_distance_scale": CENTER_DISTANCE_SCALE,
    "descriptor_duplicate_similarity": DESCRIPTOR_DUPLICATE_SIMILARITY,
    "duplicate_iou": DUPLICATE_IOU,
    "cut_gap": CUT_GAP,
    "cut_dino": CUT_DINO,
    "cut_dino_gap": CUT_DINO_GAP,
    "cut_after": CUT_AFTER,
    "bad_frame_tracks": BAD_FRAME_TRACKS,
    "switch_margin": SWITCH_MARGIN,
}


def metadata() -> dict:
    """The memory's part of the graph stage's metadata. The reliable masks depend on MIN_CONFIDENCE too."""
    return {"algorithm": "track-association-v2", "config": PARAMS, "min_confidence": MIN_CONFIDENCE}


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


def update_bank(bank, descriptor):
    e = normalized(descriptor)
    if e is None or (bank and e.shape != bank[0].shape):
        return bank
    if bank and prototype_similarity(e, bank) >= DESCRIPTOR_DUPLICATE_SIMILARITY:
        return bank
    candidates = [*bank, e.astype(np.float16)]
    if len(candidates) > MAX_DESCRIPTORS:
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
    reliable: bool = True  # whether its depth is trusted (mask_boxes.reliable); if not, it only follows its track
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

    Merges leave gaps in the IDs; numbering() closes them. Greedy strongest-pair-first matching is deterministic and
    one-to-one within a frame. Match traces contain selected/best candidate scores before updates.
    """

    def __init__(self):
        self.nodes: dict[int, ObjectNode] = {}
        self.next_id = 0
        self.last_frame: int | None = None
        self.last_matches: list[dict] = []
        self.last_merges: list[dict] = []
        self.dino_dimension: int | None = None
        self.siglip_dimension: int | None = None
        self.tracks: dict[int, int] = {}  # SAM ID -> the node its masks join
        self.track_last: dict[int, tuple[np.ndarray | None, Box3D]] = {}  # SAM ID -> DINO, box of last reliable mask
        self.merged: dict[int, int] = {}  # node ID -> the node it was merged into
        self.jumps: dict[int, int] = {}  # SAM ID -> reliable masks in a row it jumped in, while under CUT_AFTER

    def find(self, object_id: int) -> int:
        """The node an ID stands for now, after merges."""
        while object_id in self.merged:
            object_id = self.merged[object_id]
        return object_id

    def components(self, observation: Observation, node: ObjectNode) -> dict:
        return self._pair([observation.dino], observation.box, node)

    def _pair(self, descriptors, box: Box3D | None, node: ObjectNode) -> dict:
        """The match scores of an observation's, or another node's, descriptors and box against a node.

        compatible is False for a node over MAX_CANDIDATE_DISTANCE away, which scores 0.
        """
        empty = {
            "dino": 0.0,
            "inside": 0.0,
            "center": 0.0,
            "distance": None,
            "geometry": 0.0,
            "score": 0.0,
            "compatible": False,
        }
        distance = None
        inside = center = 0.0
        if valid_box(box) and valid_box(node.box):
            distance = float(np.linalg.norm(box.center - node.box.center))
            if distance > MAX_CANDIDATE_DISTANCE:
                return {**empty, "distance": distance}
            inside = float(box_containment(box, node.box))
            inside = float(np.clip(inside, 0, 1)) if np.isfinite(inside) else 0.0
            center = float(np.exp(-distance / CENTER_DISTANCE_SCALE))
        dino = max((prototype_similarity(d, node.descriptors) for d in descriptors), default=0.0)
        geometry = GEOMETRY_INSIDE_WEIGHT * inside + (1 - GEOMETRY_INSIDE_WEIGHT) * center
        wd, wg = WEIGHTS
        score = float(wd * dino + wg * geometry)
        return {
            "dino": dino,
            "inside": inside,
            "center": center,
            "distance": distance,
            "geometry": geometry,
            "score": score,
            "compatible": True,
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

        It is with a gap over CUT_GAP, or a DINO cosine under CUT_DINO where the boxes are also over CUT_DINO_GAP apart.
        """
        return gap > CUT_GAP or (dino < CUT_DINO and gap > CUT_DINO_GAP)

    def _rival(self, observation: Observation, object_id: int) -> dict | None:
        """The other node a track's mask matches SWITCH_MARGIN better than its own, as {"own", "rival", "score"}.

        None if there is none. SAM switches between neighbours that look alike, where neither the gap nor DINO tell.
        """
        own = self.components(observation, self.nodes[object_id])["score"]  # 0 when over max_candidate_distance away
        best = None
        for other_id, other in self.nodes.items():
            if other_id != object_id and other.reliable:
                score = self.score(observation, other)
                if score > THRESHOLD and score > own + SWITCH_MARGIN and (best is None or score > best[1]):
                    best = (other_id, score)
        return None if best is None else {"own": own, "rival": best[0], "score": best[1]}

    def associate_frame(self, frame: int, observations: list[Observation]) -> np.ndarray:
        if self.last_frame is not None and frame <= self.last_frame:
            raise ValueError("frames must be strictly increasing; submit all observations together")
        self.last_merges = []
        # SAM's tracks first: a mask whose track has a node joins it, unless the track jumped to another object
        follow = {}
        for row, observation in enumerate(observations):
            nodes = sorted({self.find(self.tracks[s]) for s in observation.sam_ids if s in self.tracks})
            if nodes:
                follow[row] = nodes
        moves = {row: self._move(observations[row]) for row in follow}
        moved = [m for m in moves.values() if m is not None]
        jumped = sum(gap > CUT_GAP for _, gap in moved)
        bad_pose = jumped >= BAD_FRAME_TRACKS and 2 * jumped >= len(moved)
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
                    if jumps >= CUT_AFTER:
                        for s in sam_ids:
                            self.jumps.pop(s, None)
                        cuts[row] = {"dino": move[0], "gap": move[1]}
                        continue
                    # Not cut yet (CUT_AFTER): it stays with the node, but only as its member, like an unreliable mask
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
                if parts["score"] > THRESHOLD:
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
            parts = self._pair(node.descriptors, node.box, other)
            if not parts["compatible"] or parts["score"] <= THRESHOLD:
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
            keep.descriptors = update_bank(keep.descriptors, descriptor)
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

    def numbering(self) -> dict[int, int]:
        """The nodes numbered 0, 1, ... in order of first appearance, closing the gaps merges left.

        Returns the number of every ID given so far, a merged one's being that of the node it went into.
        """
        new = {old: k for k, old in enumerate(sorted(self.nodes))}
        return {old: new[self.find(old)] for old in range(self.next_id)}

    def _update(self, node: ObjectNode, observation: Observation, frame: int, track: bool = True) -> None:
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
            node.descriptors = update_bank(node.descriptors, dino)
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
