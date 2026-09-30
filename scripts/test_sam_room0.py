"""Test SAM 1 ViT-B automatic masks on one Replica room_0 frame.

Run from the project directory:
    micromamba run -n eai-room0 python scripts/test_sam_room0.py
"""

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from utils import replica


def main() -> None:
    checkpoint = replica.REPO_ROOT.parent / "openlex3d_data" / "checkpoints" / "sam_vit_b_01ec64.pth"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--seq", choices=["00", "01"], default="00")
    parser.add_argument("--checkpoint", type=Path, default=checkpoint)
    parser.add_argument("--output", type=Path, help="Where to save the comparison figure")
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening a window")
    args = parser.parse_args()
    if not 0 <= args.frame < replica.N_FRAMES:
        parser.error(f"frame must be between 0 and {replica.N_FRAMES - 1}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}. Set --checkpoint to the ViT-B checkpoint.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable. Run with the eai-room0 environment on your ROCm machine.")

    # ROCm PyTorch uses the same 'cuda' device interface as NVIDIA PyTorch.
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    image, depth, instance_ids = replica.load_frame("room_0", args.seq, args.frame)
    print("Image:", image.shape, "depth:", depth.shape, "instance IDs:", instance_ids.shape, flush=True)

    sam = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint))
    sam.to(device="cuda").eval()
    mask_generator = SamAutomaticMaskGenerator(sam)

    print("Generating automatic masks...", flush=True)
    started = perf_counter()
    with torch.inference_mode():
        masks = mask_generator.generate(image)
    print(f"Number of masks: {len(masks)} ({perf_counter() - started:.1f} seconds)", flush=True)
    if not masks:
        raise RuntimeError("SAM returned no masks for this frame.")
    if any(mask["segmentation"].shape != image.shape[:2] for mask in masks):
        raise RuntimeError("SAM masks do not match the original image resolution.")

    for i, mask in enumerate(masks[:10]):
        print(i, "area =", mask["area"], "bbox =", mask["bbox"], "score =", mask["predicted_iou"])

    # Draw large masks first, then let smaller masks overwrite their colours.
    # One overlay keeps memory use small; seeded colours make reruns comparable.
    rng = np.random.default_rng(0)
    overlay = np.zeros((*image.shape[:2], 4), dtype=np.float32)
    for mask in sorted(masks, key=lambda item: item["area"], reverse=True):
        overlay[mask["segmentation"]] = [*rng.random(3), 0.45]

    # Ground-truth instance IDs and SAM regions are separate sets of masks.
    colours = np.random.default_rng(0).uniform(0.2, 1.0, (int(instance_ids.max()) + 1, 3))
    colours[0] = 0
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout="constrained")
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("RGB")
    depth_plot = axes[0, 1].imshow(np.ma.masked_equal(depth, 0), cmap="turbo")
    axes[0, 1].set_title("Depth")
    fig.colorbar(depth_plot, ax=axes[0, 1], label="metres", shrink=0.8)
    axes[1, 0].imshow(colours[instance_ids])
    axes[1, 0].set_title("Replica instance IDs (ground truth)")
    axes[1, 1].imshow(image)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f"SAM 1 ViT-B: {len(masks)} automatic masks")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"room_0 / sequence {args.seq} / frame {args.frame}")

    output = args.output or replica.REPO_ROOT / "outputs" / "sam_room0" / f"{args.seq}_{args.frame:05d}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print("Saved comparison:", output, flush=True)
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
