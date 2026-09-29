"""Run Hugging Face SAM 3 video tracking on a short RGB clip and save masks to disk."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor

MODEL_ID = "facebook/sam3"


@torch.inference_mode()
def segment_clip(frame_paths: list[Path], prompt: str, output_dir: Path) -> Path:
    """Save one NPZ per frame, plus a manifest mapping clip indices to source images.

    Masks have the original image resolution. Object IDs are SAM tracks within this clip,
    not ground-truth instances or persistent graph IDs. Reusing output_dir overwrites this run.
    Requires access to facebook/sam3 and a Hugging Face login on the machine running inference.
    """
    if not frame_paths or not prompt.strip():
        raise ValueError("Provide at least one frame and a non-empty text prompt.")
    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3 requires CUDA here. Select the project's GPU environment/kernel.")

    frames = []
    for path in frame_paths:
        with Image.open(path) as image:
            frames.append(np.array(image.convert("RGB")))
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("All clip frames must have the same resolution.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": MODEL_ID,
        "prompt": prompt,
        "frames": [str(Path(path).resolve()) for path in frame_paths],
        "complete": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Loading {MODEL_ID}; tracking {prompt!r} over {len(frames)} frames.")
    processor = Sam3VideoProcessor.from_pretrained(MODEL_ID)
    model = Sam3VideoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    session = processor.init_video_session(
        video=frames,
        inference_device="cuda",
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    processor.add_text_prompt(session, text=prompt)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for prediction in model.propagate_in_video_iterator(session, show_progress_bar=True):
            result = processor.postprocess_outputs(session, prediction)
            np.savez_compressed(
                output_dir / f"{prediction.frame_idx:05d}.npz",
                masks=result["masks"].cpu().numpy(),
                object_ids=result["object_ids"].cpu().numpy(),
                scores=result["scores"].float().cpu().numpy(),
                boxes=result["boxes"].float().cpu().numpy(),
            )

    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved predictions to {output_dir}")
    return output_dir


def main() -> None:
    from utils import replica

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="room_0", choices=replica.SCENES)
    parser.add_argument("--seq", default="00", choices=["00", "01"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--prompt", default="lamp")
    parser.add_argument("--output", type=Path, default=replica.REPO_ROOT / "outputs" / "sam3_preview")
    args = parser.parse_args()
    if args.start < 0 or args.frames < 1 or args.step < 1:
        parser.error("start must be non-negative; frames and step must be positive")
    rgb_dir = replica.sequence_dir(args.scene, args.seq) / "rgb"
    paths = [rgb_dir / f"rgb_{args.start + i * args.step}.png" for i in range(args.frames)]
    segment_clip(paths, args.prompt, args.output)


if __name__ == "__main__":
    main()
