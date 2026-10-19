"""Extract masks from a video with SAM 3, prompted by the classes you want to segment."""

from dataclasses import replace
from functools import cache
from types import SimpleNamespace

import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.models.sam3_video import modeling_sam3_video

from data.stages import Masks, encode_rle

MODEL_ID = "facebook/sam3"
STATE_DEVICE: str | None = None  # override the automatic tracker state device with "cpu" or "cuda"
DETECT_BATCH: int | None = None  # override the automatic prompt batch; 1 runs them one by one as SAM 3 does


def memory_settings() -> tuple[str, int]:
    """Use CPU tracking state and single-prompt detection on ROCm GPUs with at most 16 GiB of VRAM."""
    low_memory = torch.version.hip is not None and torch.cuda.get_device_properties(0).total_memory <= 16 * 2**30
    state = STATE_DEVICE if STATE_DEVICE is not None else ("cpu" if low_memory else "cuda")
    batch = DETECT_BATCH if DETECT_BATCH is not None else (1 if low_memory else 16)
    return state, batch


@cache
def load_model() -> tuple[Sam3VideoProcessor, Sam3VideoModel]:
    """Load SAM 3 onto the GPU once per process; later calls return the same model."""
    state, batch = memory_settings()
    print(f"SAM 3: tracker state on {state}, detection batch {batch}")
    processor = Sam3VideoProcessor.from_pretrained(MODEL_ID)
    model = Sam3VideoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    model._tracker_update_memories = offload_memories(model._tracker_update_memories, model.tracker_model)
    tracker = model.tracker_model
    if state == "cpu":
        tracker._encode_new_memory = chunked_memory_encoding(tracker._encode_new_memory)
        modeling_sam3_video.mask_iou = mask_iou
    tracker._batch_encode_memories = offload_memories(tracker._batch_encode_memories, tracker)
    model.run_detection = batched_detection(model)
    return processor, model


def mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """Pairwise binary-mask IoU without allocating an (N, M, pixels) intermediate.

    Binary intersections are dot products; float32 counts stay exact at SAM's mask resolution. Disable autocast so
    bf16 does not round large areas and change association decisions. Memory grows with (N + M) * pixels instead.
    """
    with torch.autocast(pred_masks.device.type, enabled=False):
        pred, gt = pred_masks.flatten(1).float(), gt_masks.flatten(1).float()
        intersection = pred @ gt.T
        union = pred.sum(1, keepdim=True) + gt.sum(1)[None] - intersection
        return intersection / union.clamp_min(1)


def chunked_memory_encoding(encode, batch_size: int = 16):
    """Encode object memories in bounded batches, preserving the object axis in both returned tensors.

    Both tracker passes encode every tracked object together. Their mask upsampling and memory fuser can allocate
    several GiB at once even when stored memories live on the CPU. The encoder processes objects independently, so
    splitting that batch bounds its temporary memory without changing which objects are tracked.
    """

    def wrapped(current_vision_feats, pred_masks_high_res, object_score_logits, is_mask_from_pts):
        count = len(pred_masks_high_res)
        if count <= batch_size:
            return encode(current_vision_feats, pred_masks_high_res, object_score_logits, is_mask_from_pts)
        features, positions = [], []
        for start in range(0, count, batch_size):
            stop = start + batch_size
            # One shared image feature may be broadcast across every object's mask.
            vision = current_vision_feats if current_vision_feats.shape[1] == 1 else current_vision_feats[:, start:stop]
            feat, pos = encode(
                current_vision_feats=vision,
                pred_masks_high_res=pred_masks_high_res[start:stop],
                object_score_logits=object_score_logits[start:stop],
                is_mask_from_pts=is_mask_from_pts,
            )
            features.append(feat)
            positions.append(pos)
        return torch.cat(features, dim=1), torch.cat(positions, dim=1)

    return wrapped


def batched_detection(model: Sam3VideoModel):
    """SAM 3's run_detection, batching prompts according to memory_settings instead of always running one by one.

    In every frame, transformers 5.17 runs the image encoder once and then the detector once for each prompt, which
    with 102 prompts is most of SAM 3's time. Every prompt is padded to the same 32 tokens, so a chunk of prompts is
    one batch against the frame's image features, and each prompt's detections are kept as run_detection keeps them.
    """
    original = model.run_detection

    def run_detection(inference_session, vision_embeds):
        _, detect_batch = memory_settings()
        if detect_batch <= 1:
            return original(inference_session=inference_session, vision_embeds=vision_embeds)
        detector = model.detector_model
        prompt_ids = list(inference_session.prompts.keys())
        for prompt_id in prompt_ids:
            if prompt_id not in inference_session.prompt_embeddings:
                inference_session.prompt_embeddings[prompt_id] = detector.get_text_features(
                    input_ids=inference_session.prompt_input_ids[prompt_id],
                    attention_mask=inference_session.prompt_attention_masks[prompt_id],
                    return_dict=True,
                )
        detections = {}
        for start in range(0, len(prompt_ids), detect_batch):
            chunk = prompt_ids[start : start + detect_batch]
            n = len(chunk)
            # The detector reads only pooler_output, which get_text_features sets on its output
            text = SimpleNamespace(
                pooler_output=torch.cat([inference_session.prompt_embeddings[p].pooler_output for p in chunk])
            )
            vision = type(vision_embeds)(
                fpn_hidden_states=tuple(x.expand(n, *x.shape[1:]) for x in vision_embeds.fpn_hidden_states),
                fpn_position_encoding=tuple(x.expand(n, *x.shape[1:]) for x in vision_embeds.fpn_position_encoding),
            )
            attention = torch.cat([inference_session.prompt_attention_masks[p] for p in chunk])
            out = detector(vision_embeds=vision, text_embeds=text, attention_mask=attention)
            probs = out.pred_logits.sigmoid() * out.presence_logits.sigmoid()
            for k, prompt_id in enumerate(chunk):
                pred_probs = probs[k : k + 1]
                if model.det_nms_thresh > 0.0:
                    keep = modeling_sam3_video.nms_masks(
                        pred_probs=pred_probs[0],
                        pred_masks=out.pred_masks[k],
                        prob_threshold=model.score_threshold_detection,
                        iou_threshold=model.det_nms_thresh,
                    )
                    pred_probs = pred_probs.clone()
                    pred_probs[0][~keep] = 0.0
                pos = torch.where(pred_probs > model.score_threshold_detection)
                detections[prompt_id] = {
                    "bbox": out.pred_boxes[k : k + 1][pos[0], pos[1]],
                    "mask": out.pred_masks[k : k + 1][pos[0], pos[1]],
                    "scores": pred_probs[pos[0], pos[1]],
                }
        return detections

    return run_detection


def offload_memories(encode, tracker):
    """Wrap one of SAM 3's two memory encoders so the memories it stores go to the state's device, and old ones go.

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
    The whole clip stays in CPU memory, and memory_settings chooses the tracker's state device. Without forget, the
    tracker kept every frame's masks and features for every object and outgrew 32 GB of GPU at 100 frames; with it,
    the state is the last
    few frames of every object found, and keeping it on the CPU instead cost about 40% of the speed.
    """
    if len(frames) != len(frame_numbers):
        raise ValueError(f"got {len(frames)} frames but {len(frame_numbers)} frame numbers")
    if len(set(prompts)) != len(prompts):
        raise ValueError("prompts must be distinct: the session merges duplicates")

    state_device, _ = memory_settings()
    inference_session = processor.init_video_session(
        video=frames,
        inference_device="cuda",
        inference_state_device=state_device,
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
