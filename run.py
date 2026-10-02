"""Run the pipeline on Replica runs. Each stage caches its output, so reruns only compute what is missing."""

import argparse

from models import sam
from utils import replica
from utils.schema import Run

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
    if run.mask_source == "gt":
        replica.save_gt_masks(run, args.overwrite)
    else:
        sam.save_masks(run, args.overwrite)
