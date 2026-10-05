"""Extract masks from a video with SAM 3, prompted by the classes you want to segment."""

from functools import cache

import numpy as np
import pycocotools.mask as mask_utils
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

from utils.schema import Masks, encode_rle

MODEL_ID = "facebook/sam3"


@cache
def load_model() -> tuple[Sam3VideoProcessor, Sam3VideoModel]:
    """Load SAM 3 onto the GPU once per process; later calls return the same model."""
    processor = Sam3VideoProcessor.from_pretrained(MODEL_ID)
    model = Sam3VideoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    return processor, model


@torch.inference_mode()
def segment(
    frames: list[np.ndarray],
    frame_numbers: list[int],
    prompts: list[str],
    processor: Sam3VideoProcessor,
    model: Sam3VideoModel,
) -> Masks:
    """Track every prompt through the frames in one video session, so SAM IDs are unique across prompts.

    frames are RGB (H, W, 3) uint8 in temporal order, and Masks.frame holds frame_numbers[i] for frames[i].
    Masks.prompt indexes prompts. SAM 3 keeps the masks of one prompt apart, but masks of different prompts can overlap.
    The whole clip and the tracker state stay in CPU memory during tracking. The tracker keeps every frame's masks and
    features for every object, so on the GPU it outgrew 32 GB at 100 frames; on the CPU it costs about 40% speed.
    """
    if len(frames) != len(frame_numbers):
        raise ValueError(f"got {len(frames)} frames but {len(frame_numbers)} frame numbers")
    if len(set(prompts)) != len(prompts):
        raise ValueError("prompts must be distinct: the session merges duplicates")

    inference_session = processor.init_video_session(
        video=frames,
        inference_device="cuda",
        inference_state_device="cpu",
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    processor.add_text_prompt(inference_session, prompts)
    prompt_index = {p: i for i, p in enumerate(prompts)}

    frame, prompt, sam_id, score, rle = [], [], [], [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for model_prediction in model.propagate_in_video_iterator(inference_session, show_progress_bar=True):
            result = processor.postprocess_outputs(inference_session, model_prediction)
            ids = result["object_ids"].tolist()
            if not ids:
                continue
            prompt_of = {i: prompt_index[p] for p, objs in result["prompt_to_obj_ids"].items() for i in objs}
            # frame_idx counts frames within this clip; encode right away so full-size masks never pile up
            frame += [frame_numbers[model_prediction.frame_idx]] * len(ids)
            prompt += [prompt_of[i] for i in ids]
            sam_id += ids
            score += result["scores"].tolist()
            rle += encode_rle(result["masks"].cpu().numpy())

    return Masks(
        frame=np.array(frame, dtype=np.int32),
        prompt=np.array(prompt, dtype=np.int16),
        sam_id=np.array(sam_id, dtype=np.int32),
        score=np.array(score, dtype=np.float32),
        rle=rle,
    )


def mask_boxes(rle: list[dict]) -> np.ndarray:
    """(N, 4) int32 bounding boxes (x0, y0, x1, y1) of Masks.rle rows, so image[y0:y1, x0:x1] crops mask i.

    An empty mask gets the box (0, 0, 0, 0), so check x1 > x0 before cropping.
    """
    # toBbox gives COCO (x, y, w, h) with integer values, and a flat array when rle is empty
    xywh = mask_utils.toBbox(rle).reshape(-1, 4).astype(np.int32)
    return np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], axis=1)
