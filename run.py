"""Run the pipeline on Replica runs. Each stage caches its output, so reruns only compute what is missing."""

import argparse

from models import sam
from utils import replica
from utils.schema import Run, save_stage

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--source",
    nargs="+",
    default=[f"replica/{scene}/00" for scene in replica.SCENES],
    help="runs to compute, such as replica/room_0/00 (default: every scene's sequence 00)",
)
parser.add_argument("--stride", type=int, default=20)
parser.add_argument("--mask-source", default="sam3", choices=["sam3", "gt"])
parser.add_argument("--pose-source", default="gt", choices=["gt", "vggt"])
parser.add_argument("--overwrite", action="store_true", help="recompute stages that are already cached")
args = parser.parse_args()
if args.stride < 1:
    parser.error("stride must be positive")

for source in args.source:
    run = Run(source=source, stride=args.stride, mask_source=args.mask_source, pose_source=args.pose_source)
    if run.done("masks") and not args.overwrite:
        print(f"Skipping {run.source}: {run.stage_dir('masks')} exists")
        continue

    if run.mask_source == "gt":
        masks, vocab = replica.gt_masks(run)
        metadata = {"vocab": vocab}
    else:
        frames, numbers = replica.load_run_rgb(run)
        vocab = replica.prompt_vocab(run)
        processor, model = sam.load_model()
        masks = sam.segment(frames, numbers, vocab, processor, model)
        metadata = {
            "vocab": vocab,
            "model": sam.MODEL_ID,
            "score_threshold_detection": model.config.score_threshold_detection,
            "new_det_thresh": model.config.new_det_thresh,
        }
    save_stage(run.stage_dir("masks"), masks, metadata)
    print(f"Saved {len(masks.frame)} masks to {run.stage_dir('masks')}")
