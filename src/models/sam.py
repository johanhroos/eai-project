"""Extract masks from a video with SAM 3, prompted by the classes you want to segment."""

from dataclasses import replace
from functools import cache

import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

from data.stages import Masks, encode_rle

MODEL_ID = "facebook/sam3"


@cache
def load_model() -> tuple[Sam3VideoProcessor, Sam3VideoModel]:
    """Load SAM 3 onto the GPU once per process; later calls return the same model."""
    processor = Sam3VideoProcessor.from_pretrained(MODEL_ID)
    model = Sam3VideoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    model._tracker_update_memories = offload_memories(model._tracker_update_memories, model.tracker_model)
    tracker = model.tracker_model
    tracker._batch_encode_memories = offload_memories(tracker._batch_encode_memories, tracker)
    return processor, model


def offload_memories(encode, tracker):
    """Wrap one of SAM 3's two memory encoders so the memories it stores go to the CPU, and old ones are dropped.

    transformers 5.17 moves the tracker's outputs to inference_state_device but stores the encoded memories, about
    1.3 MB per object and frame, where they were computed. On the GPU they took 13.7 GB by frame 139 of IMG_0649 with
    7 prompts (90 objects), and the 249 frames ran out of 32 GB. The tracker moves each memory back to the GPU when it
    attends to it. On the CPU, it keeps every frame's outputs for every object: with the masks it stores too, 3.5 MB per
    object and frame, 35 GB of RAM after 150 frames of IMG_0649 (89 objects), and its 249 frames ran out of 60 GB.
    forget drops what the tracker never reads again: the same 150 frames then take 4.7 GB, with the same masks.
    """

    def wrapped(inference_session, frame_idx, **kwargs):
        encode(inference_session=inference_session, frame_idx=frame_idx, **kwargs)
        for outputs in inference_session.output_dict_per_obj.values():
            for stored in outputs.values():
                out = stored.get(frame_idx)
                for key in ("maskmem_features", "maskmem_pos_enc"):
                    # Blocking: a non_blocking copy to the CPU can return before it is filled
                    if out is not None and isinstance(out.get(key), torch.Tensor):
                        out[key] = out[key].to(inference_session.inference_state_device)
            forget(outputs, frame_idx, tracker.num_maskmem, tracker.config.max_cond_frame_num)

    return wrapped


def forget(outputs: dict, frame_idx: int, num_maskmem: int, max_cond_frames: int) -> None:
    """Drop what SAM 3's tracker never reads again from one object's stored outputs, once frame_idx is encoded.

    Tracking frame f (always forward here), the tracker reads the memories of frames f - 1 to f - (num_maskmem - 1) and
    of the max_cond_frames conditioning frames closest to f, and the object pointers (tiny, kept) of up to 16 frames.
    It reads the masks it stores only in the frame they are from. So the masks of every earlier frame go, and so do the
    memories of the frames before that window, except those of the last max_cond_frames conditioning frames.
    """
    oldest = frame_idx - (num_maskmem - 1)  # the oldest frame the next frame's memory window reaches, and one more
    newest_conditioning = sorted(outputs["cond_frame_outputs"])[-max_cond_frames:]
    for kind, stored in outputs.items():
        for frame, out in stored.items():
            if frame >= frame_idx:
                continue
            out.pop("high_res_masks", None)
            out.pop("pred_masks", None)
            if frame < oldest and not (kind == "cond_frame_outputs" and frame in newest_conditioning):
                out.pop("maskmem_features", None)
                out.pop("maskmem_pos_enc", None)


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
    The whole clip and the tracker state stay in CPU memory during tracking, memories included (offload_memories). The
    tracker keeps every frame's masks and features for every object, so on the GPU it outgrew 32 GB at 100 frames; on
    the CPU it costs about 40% speed.
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


def segment_clips(
    frames: list[np.ndarray],
    frame_numbers: list[int],
    clip: np.ndarray,
    prompts: list[str],
    processor: Sam3VideoProcessor,
    model: Sam3VideoModel,
) -> Masks:
    """Track each video of a run on its own with segment, and make the SAM IDs unique across them.

    clip gives the video of every frame (footage.clip_index). Tracking across the cut between two videos would carry an
    ID over to whatever object is at the same place in the next video's first frame; linking objects across videos is
    the graph's job.
    """
    parts, offset = [], 0
    for k in np.unique(clip):
        sel = np.flatnonzero(clip == k).tolist()
        part = segment([frames[i] for i in sel], [frame_numbers[i] for i in sel], prompts, processor, model)
        parts.append(replace(part, sam_id=part.sam_id + offset))
        offset += int(part.sam_id.max()) + 1 if len(part.sam_id) else 0
    return Masks(
        frame=np.concatenate([m.frame for m in parts]),
        prompt=np.concatenate([m.prompt for m in parts]),
        sam_id=np.concatenate([m.sam_id for m in parts]),
        score=np.concatenate([m.score for m in parts]),
        rle=[r for m in parts for r in m.rle],
    )
