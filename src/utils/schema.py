import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils

from utils.config import OUTPUT_DIR, STAGE_KEYS


def _check(name: str, a, dtype, shape: tuple) -> None:
    """Raise if `a` is not an array with this dtype and shape. None in `shape` matches any size."""
    if not isinstance(a, np.ndarray):
        raise TypeError(f"{name}: expected np.ndarray, got {type(a).__name__}")
    if a.dtype != dtype:
        raise TypeError(f"{name}: expected dtype {np.dtype(dtype)}, got {a.dtype}")
    if a.ndim != len(shape):
        raise ValueError(f"{name}: expected {len(shape)} dimensions, got shape {a.shape}")
    for expected, actual in zip(shape, a.shape, strict=True):
        if expected is not None and expected != actual:
            raise ValueError(f"{name}: expected shape {shape}, got {a.shape}")


@dataclass(frozen=True)
class Run:
    source: str  # "replica/room_0/00" or "phone/kitchen_1"
    stride: int
    mask_source: str  # "sam3" | "gt"
    pose_source: str  # "gt" | "vggt"
    vocab: str = ""  # name of the SAM prompt list given with run.py --vocab, "" for the source's default

    def stage_dir(self, stage: str) -> Path:
        tag = "_".join(getattr(self, k) for k in STAGE_KEYS[stage] if getattr(self, k))
        return OUTPUT_DIR / self.source / f"stride{self.stride}" / tag / stage

    def done(self, stage: str) -> bool:
        return (self.stage_dir(stage) / "metadata.json").exists()


@dataclass
class Features:
    """Appearance embedding of every mask: row i belongs to mask i of the masks stage."""

    dino: np.ndarray  # (N, D) float16, L2-normalized, zero for an empty mask

    def __post_init__(self):
        n = len(self.dino)
        _check("dino", self.dino, np.float16, (n, None))


@dataclass
class TextAligned:
    """SigLIP image embedding of every mask, comparable with SigLIP text embeddings: row i belongs to mask i."""

    siglip: np.ndarray  # (N, D) float16, L2-normalized, zero for an empty mask

    def __post_init__(self):
        _check("siglip", self.siglip, np.float16, (len(self.siglip), None))


@dataclass
class Masks:
    """SAM 3 output for one set of frames. Row i of every field describes mask i. N masks in total."""

    frame: np.ndarray  # (N,) int32, frame number
    prompt: np.ndarray  # (N,) int16, index into metadata["vocab"]
    sam_id: np.ndarray  # (N,) int32, SAM object ID
    score: np.ndarray  # (N,) float32, SAM predicted confidence
    # pycocotools RLE (https://en.wikipedia.org/wiki/Run-length_encoding) format for efficient storage of masks
    # can also efficiently compute e.g. IoU
    rle: list[dict]  # N RLE dicts: {'size', 'counts'}

    def __post_init__(self):
        n = len(self.frame)
        _check("frame", self.frame, np.int32, (n,))
        _check("prompt", self.prompt, np.int16, (n,))
        _check("sam_id", self.sam_id, np.int32, (n,))
        _check("score", self.score, np.float32, (n,))
        if len(self.rle) != n:
            raise ValueError(f"rle: got {len(self.rle)} rows, expected {n}")
        if self.rle and not isinstance(self.rle[0]["counts"], str):
            raise TypeError("rle counts must be str: build rle with encode_rle")


def encode_rle(masks: np.ndarray) -> list[dict]:
    """Masks.rle rows for a (N, H, W) bool stack of masks."""
    # pycocotools wants a Fortran-ordered (H, W, N) uint8 stack and returns counts as bytes, which JSON can't hold
    rles = mask_utils.encode(masks.transpose(1, 2, 0).astype(np.uint8, order="F"))
    return [{"size": r["size"], "counts": r["counts"].decode("ascii")} for r in rles]


def check_aligned(run: Run, stage: str) -> None:
    """Check that all stages operate on the same number of masks."""
    n = json.loads((run.stage_dir(stage) / "metadata.json").read_text())["n_rows"]
    n_masks = json.loads((run.stage_dir("masks") / "metadata.json").read_text())["n_rows"]
    if n != n_masks:
        raise ValueError(f"{stage} has {n} rows, masks has {n_masks}")


@dataclass
class Graph:
    """Persistent node of every mask, row-aligned with masks (graph/pipeline.py).

    Masks join the node of their SAM track, one frame at a time. A track that jumps to another object is cut, and new
    tracks are matched to the nodes by appearance and location, so several tracks can end up in one node and, where
    SAM's ID switched objects, one track in several. A prefix replay (graph/visualize.py) uses -1 for the rows not
    processed yet.
    """

    node: np.ndarray  # (N,) int32, node ID: 0, 1, ... in order of first appearance
    # (M, 3) float32 by node ID, world frame: each node's box from every view of it (graph/shape.py). None in a graph
    # that only replays the memory, whose boxes graph.node.node_box draws around the pooled points
    box_min: np.ndarray | None = None
    box_max: np.ndarray | None = None
    # (M,) int16 by node ID: each node's label, an index into the masks stage's vocab, from all its views
    # (graph/refine.py labels); -1 for IDs no mask has. None: graph.node.build_nodes counts SAM's votes
    label: np.ndarray | None = None

    def __post_init__(self):
        _check("node", self.node, np.int32, (len(self.node),))
        if (self.box_min is None) != (self.box_max is None):
            raise ValueError("box_min and box_max: give both or neither")
        n_ids = int(self.node.max()) + 1 if len(self.node) else 0
        if self.box_min is not None:
            _check("box_min", self.box_min, np.float32, (n_ids, 3))
            _check("box_max", self.box_max, np.float32, (n_ids, 3))
        if self.label is not None:
            _check("label", self.label, np.int16, (n_ids,))


@dataclass
class Frames:
    """The frames of a phone run: every stride-th frame of its videos, saved as images next to this stage's arrays.

    A run of several videos numbers their frames one after the other, as if the videos were joined.
    """

    frame: np.ndarray  # (F,) int32, frame numbers
    height: int  # of the saved images, after rotation and downscaling
    width: int
    fps: float  # of the first video
    clip: np.ndarray | None = None  # (F,) int16, which video each frame is from (None: one video)
    clips: list[str] | None = None  # the videos' names, in order

    def __post_init__(self):
        _check("frame", self.frame, np.int32, (len(self.frame),))
        if self.clip is not None:
            _check("clip", self.clip, np.int16, (len(self.frame),))


@dataclass
class Geometry:
    """Depth and camera of every frame of a run from VGGT-Omega, in the model's units and world frame (models/vggt.py).

    sim3 maps that world frame to one in metres with z up, so its scale also converts depth and translations to metres.
    """

    frame: np.ndarray  # (F,) int32, frame numbers
    depth: np.ndarray  # (F, h, w) float16, z-depth at the model's input resolution, model units
    conf: np.ndarray  # (F, h, w) float16, confidence of the depth: higher is better
    intrinsics: np.ndarray  # (F, 3, 3) float32, for the full-resolution image
    cam_from_world: np.ndarray  # (F, 3, 4) float32, OpenCV camera, model units
    sim3: np.ndarray  # (4, 4) float32, [s R | t]: model world point to aligned world point in metres

    def __post_init__(self):
        n = len(self.frame)
        _check("frame", self.frame, np.int32, (n,))
        _check("depth", self.depth, np.float16, (n, None, None))
        _check("conf", self.conf, np.float16, self.depth.shape)
        _check("intrinsics", self.intrinsics, np.float32, (n, 3, 3))
        _check("cam_from_world", self.cam_from_world, np.float32, (n, 3, 4))
        _check("sim3", self.sim3, np.float32, (4, 4))

    @property
    def scale(self) -> float:
        """Metres per model unit."""
        return float(np.linalg.norm(self.sim3[:3, 0]))


@dataclass
class Events:
    """Ground-truth hide and revisit events, one row per event, in frame order. Not aligned with the masks."""

    object_id: np.ndarray  # (E,) int32, Replica object ID
    frame: np.ndarray  # (E,) int32, frame number: last visible frame (hidden) or first visible frame (revisited)
    kind: np.ndarray  # (E,) int8, 0 = hidden, 1 = revisited
    gap: np.ndarray  # (E,) int32, frames hidden before the revisit (0 for hidden events)

    def __post_init__(self):
        n = len(self.object_id)
        _check("object_id", self.object_id, np.int32, (n,))
        _check("frame", self.frame, np.int32, (n,))
        _check("kind", self.kind, np.int8, (n,))
        _check("gap", self.gap, np.int32, (n,))


@dataclass
class Boxes:
    """3D box of every mask: row i belongs to mask i of the masks stage. World frame, metres, axis-aligned.

    box_min and box_max are NaN (and n_points 0) where a mask has too little valid depth. lift3d.reliable says which
    masks' depth to trust.
    """

    box_min: np.ndarray  # (N, 3) float32
    box_max: np.ndarray  # (N, 3) float32
    n_points: np.ndarray  # (N,) int32, depth points left after outlier removal
    points: np.ndarray | None = None  # (N, P, 3) float16, a random sample of those points, NaN-padded
    conf: np.ndarray | None = None  # (N,) float32, median depth confidence under the mask, NaN without one

    def __post_init__(self):
        n = len(self.box_min)
        _check("box_min", self.box_min, np.float32, (n, 3))
        _check("box_max", self.box_max, np.float32, (n, 3))
        _check("n_points", self.n_points, np.int32, (n,))
        if self.points is not None:
            _check("points", self.points, np.float16, (n, None, 3))
        if self.conf is not None:
            _check("conf", self.conf, np.float32, (n,))


@dataclass
class ObjectBoxes:
    """Ground-truth 3D box of each object, fused over all frames. Not aligned with the masks."""

    object_id: np.ndarray  # (M,) int32, Replica object ID
    box_min: np.ndarray  # (M, 3) float32
    box_max: np.ndarray  # (M, 3) float32
    n_points: np.ndarray  # (M,) int32

    def __post_init__(self):
        n = len(self.object_id)
        _check("object_id", self.object_id, np.int32, (n,))
        _check("box_min", self.box_min, np.float32, (n, 3))
        _check("box_max", self.box_max, np.float32, (n, 3))
        _check("n_points", self.n_points, np.int32, (n,))


def save_stage(stage_dir: Path, data, metadata: dict) -> None:
    """General function to cache data e.g. SAM masks or DINO features."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "metadata.json").unlink(missing_ok=True)

    values = {f.name: getattr(data, f.name) for f in fields(data)}
    arrays = {k: v for k, v in values.items() if isinstance(v, np.ndarray)}
    other = {k: v for k, v in values.items() if k not in arrays}
    metadata = {**metadata, "n_rows": len(getattr(data, fields(data)[0].name))}

    np.savez_compressed(stage_dir / "arrays.npz", allow_pickle=False, **arrays)
    (stage_dir / "other.json").write_text(json.dumps(other))
    (stage_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_stage[T](stage_dir: Path, stage_class: type[T]) -> T:
    """Load a stage saved by save_stage; stage_class validates the data in __post_init__."""
    if not (stage_dir / "metadata.json").exists():
        raise FileNotFoundError(f"{stage_dir} is missing or incomplete")
    with np.load(stage_dir / "arrays.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    other = json.loads((stage_dir / "other.json").read_text())
    return stage_class(**arrays, **other)


@dataclass
class PersistentObjects:
    """Compact memory snapshot; no masks, images or dense scene points required."""

    object_id: np.ndarray
    box_min: np.ndarray
    box_max: np.ndarray
    n_points: np.ndarray
    dino: np.ndarray  # (n, max_descriptors, D) float16, each node's bank of DINO descriptors, zero-padded
    siglip_sum: np.ndarray  # (n, D) float32, sum of the node's observations' SigLIP embeddings, for text search
    siglip_count: np.ndarray
    first_seen: np.ndarray
    last_seen: np.ndarray
    observation_count: np.ndarray
    labels: list[str | None]
    config: dict
    next_id: int
    last_frame: int | None
    points: np.ndarray | None = None  # (n, P, 3) float16, a sample of each node's pooled reliable points, NaN-padded
    track_id: np.ndarray | None = None  # (t,) int32, SAM IDs
    track_node: np.ndarray | None = None  # (t,) int32, the node each SAM ID's masks join
    frames: list[list[int]] | None = None  # the frames each node was seen in

    def __post_init__(self):
        n = len(self.object_id)
        if self.points is not None:
            _check("points", self.points, np.float16, (n, None, 3))
        if self.track_id is not None:
            _check("track_id", self.track_id, np.int32, (len(self.track_id),))
            _check("track_node", self.track_node, np.int32, self.track_id.shape)
        _check("object_id", self.object_id, np.int32, (n,))
        for name in ("box_min", "box_max"):
            _check(name, getattr(self, name), np.float32, (n, 3))
        for name in ("n_points", "siglip_count", "first_seen", "last_seen", "observation_count"):
            _check(name, getattr(self, name), np.int64, (n,))
        _check("dino", self.dino, np.float16, (n, None, None))
        _check("siglip_sum", self.siglip_sum, np.float32, (n, None))
        if len(self.labels) != n or len(np.unique(self.object_id)) != n:
            raise ValueError("invalid persistent node IDs or labels")
        if n and self.next_id <= int(self.object_id.max()):
            raise ValueError("next_id must exceed every stored ID")
