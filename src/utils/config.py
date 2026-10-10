from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "outputs"
DATA_DIR = ROOT_DIR / "data"

STAGE_KEYS = {
    "masks": ("mask_source",),
    "features": ("mask_source",),
    "boxes": ("mask_source", "pose_source"),
    "events": (),
    "gt_boxes": (),
}
