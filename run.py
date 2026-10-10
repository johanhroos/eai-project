"""Run the pipeline on Replica runs. Each stage caches its output, so reruns only compute what is missing."""

import argparse

from models import sam
from utils import replica
from utils import lift3d
from utils.schema import Masks, Run, check_aligned, load_stage, save_stage

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
    if not run.done("masks") or args.overwrite:
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
    else: print(f"Skipping {run.source}: {run.stage_dir('masks')} exists")
    
    if not run.done("boxes") or args.overwrite:
        masks = load_stage(run.stage_dir("masks"), Masks)
        save_stage(run.stage_dir("boxes"), lift3d.compute_boxes(run, masks), {"pose_source": run.pose_source})
        check_aligned(run, "boxes")
        print(f"Saved boxes to {run.stage_dir('boxes')}")
    else: print(f"Skipping {run.source}: {run.stage_dir('boxes')} exists")

    # Ground truth for the evaluation: always from GT masks and poses, whatever mask_source and pose_source are
    if not run.done("events") or args.overwrite:
        meta = {"min_pixels": lift3d.MIN_PIXELS, "min_run": lift3d.MIN_RUN, "min_gap": lift3d.MIN_GAP}
        save_stage(run.stage_dir("events"), lift3d.compute_events(run), {**meta, "kinds": lift3d.KIND})
        print(f"Saved events to {run.stage_dir('events')}")
    else: print(f"Skipping {run.source}: {run.stage_dir('events')} exists")


    # FOR EVALUATION: always from GT masks and poses, whatever mask_source and pose_source are
    if not run.done("gt_boxes") or args.overwrite:
        save_stage(run.stage_dir("gt_boxes"), lift3d.compute_gt_boxes(run), {})
        print(f"Saved GT boxes to {run.stage_dir('gt_boxes')}")
    else: print(f"Skipping {run.source}: {run.stage_dir('gt_boxes')} exists")

