from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "outputs"
DATA_DIR = ROOT_DIR / "data"

STAGE_KEYS = {
    "masks": ("mask_source", "vocab"),
    "features": ("mask_source", "vocab"),
    "siglip": ("mask_source", "vocab"),
    "boxes": ("mask_source", "vocab", "pose_source"),
    "graph": ("mask_source", "vocab", "pose_source"),
    "association": ("mask_source", "vocab", "pose_source"),  # graph/pipeline.py, next to the graph stage
    "eval": ("mask_source", "vocab", "pose_source"),
    "events": (),
    "vggt": (),
    "frames": (),
    "gt_boxes": (),
}
