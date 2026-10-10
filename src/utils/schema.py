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

    def stage_dir(self, stage: str) -> Path:
        tag = "_".join(getattr(self, k) for k in STAGE_KEYS[stage])
        return OUTPUT_DIR / self.source / f"stride{self.stride}" / tag / stage

    def done(self, stage: str) -> bool:
        return (self.stage_dir(stage) / "metadata.json").exists()


@dataclass
class Features:
    dino: np.ndarray  # (N, D) float16, L2-normalized
    siglip: np.ndarray  # (N, D) float16, L2-normalized image embedding

    def __post_init__(self):
        n = len(self.dino)
        _check("dino", self.dino, np.float16, (n, None))
        _check("siglip", self.siglip, np.float16, (n, None))


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

    box_min and box_max are NaN (and n_points 0) where a mask has too little valid depth.
    """

    box_min: np.ndarray  # (N, 3) float32
    box_max: np.ndarray  # (N, 3) float32
    n_points: np.ndarray  # (N,) int32, depth points left after outlier removal

    def __post_init__(self):
        n = len(self.box_min)
        _check("box_min", self.box_min, np.float32, (n, 3))
        _check("box_max", self.box_max, np.float32, (n, 3))
        _check("n_points", self.n_points, np.int32, (n,))

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
