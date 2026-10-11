"""Test DINOv3 descriptors on saved SAM3 object masks."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from utils import replica

from models.dino import DINOEncoder
from utils.image_crop import crop_bboxes


def main():
    output_dir = Path("outputs/sam3_preview")

    # Load the SAM3 manifest
    manifest = json.loads(
        (output_dir / "manifest.json").read_text()
    )

    # # Load first RGB frame
    # rgb = np.array(
    #     Image.open(manifest["frames"][0]).convert("RGB")
    # )

    rgb_path = (
        replica.sequence_dir("room_0", "00")
        / "rgb"
        / Path(manifest["frames"][0]).name
    )

    print("Loading RGB from:", rgb_path)

    rgb = np.array(
        Image.open(rgb_path).convert("RGB")
    )

    # Load first frame's SAM3 predictions
    with np.load(output_dir / "00000.npz") as data:
        masks = data["masks"]
        object_ids = data["object_ids"]

    print("RGB shape:", rgb.shape)
    print("Masks shape:", masks.shape)
    print("Object IDs:", object_ids)

    # Derive inclusive bounding boxes from masks
    bboxes = []
    valid_ids = []

    for object_id, mask in zip(object_ids, masks):
        mask = np.squeeze(mask).astype(bool)

        ys, xs = np.nonzero(mask)

        if len(xs) == 0:
            continue

        bbox = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        )

        bboxes.append(bbox)
        valid_ids.append(object_id)

    print("Valid objects:", len(bboxes))

    if not bboxes:
        print("No objects detected in this frame.")
        return

    # Load DINOv3
    dino = DINOEncoder(
        "facebook/dinov3-vitl16-pretrain-lvd1689m"
    )

    # Test the complete integration function
    embeddings, valid = dino.encode_objects(
        rgb,
        bboxes,
        padding=0.1,
        inclusive=True,
    )

    # Verify output
    print("Embeddings shape:", embeddings.shape)
    print("Valid descriptors:", valid)
    print("Embedding norms:", torch.linalg.vector_norm(embeddings, dim=1))

    assert embeddings.shape == (len(bboxes), 1024)
    assert valid.all().item()
    assert torch.allclose(
        torch.linalg.vector_norm(embeddings, dim=1),
        torch.ones(len(bboxes), device=embeddings.device),
        atol=1e-5,
    )

    for object_id, embedding in zip(valid_ids, embeddings):
        print(f"Object {object_id}: {embedding.shape}")

    print("DINO integration test passed!")


if __name__ == "__main__":
    main()
