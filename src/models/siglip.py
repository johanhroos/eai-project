"""SigLIP 2 image embeddings of object masks and text embeddings of queries, in one space: compare them by cosine."""

from functools import cache

import numpy as np
import pycocotools.mask as mask_utils
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from models.sam import mask_boxes
from utils import footage
from utils.image_crop import pad_bbox, square_bbox
from utils.schema import Masks, Run, TextAligned

MODEL_ID = "google/siglip2-so400m-patch16-384"
TEMPLATE = "This is a photo of {}."  # the prompt format of the SigLIP 2 model card
TEXT_LENGTH = 64  # SigLIP is trained on text padded to 64 tokens; shorter padding gives worse embeddings
BATCH = 32  # masks encoded at once (two crops each): a frame with all its masks at once overflows a 10 GB GPU


class SiglipEncoder:
    def __init__(self, model_name: str = MODEL_ID, device="cuda"):
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.float16).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    @torch.inference_mode()
    def encode_masks(self, image: np.ndarray, masks: np.ndarray, boxes: np.ndarray, padding=0.1) -> torch.Tensor:
        """One L2-normalized embedding per mask, (k, D) float32 on the device. An empty mask gets a zero vector.

        image is RGB (H, W, 3) uint8, masks (k, H, W) bool and boxes their (k, 4) x0, y0, x1, y1 from mask_boxes.
        Each box is padded and grown to a square, so the processor's square resize doesn't squash the object. The crop
        is encoded twice, as it is and with grey outside the mask, and the two are averaged: the first shows the object
        in context, the second which object is meant. On Replica GT masks, the average names the class of 70% of the
        objects against 54-69% for one crop alone.
        """
        empty = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
        crops, greyed = [], []
        for mask, box, skip in zip(masks, boxes, empty, strict=True):
            x0, y0, x1, y1 = (0, 0, 1, 1) if skip else square_bbox(pad_bbox(box, image.shape, padding), image.shape)
            crop = image[y0:y1, x0:x1].copy()  # torch warns about the read-only arrays PIL gives
            crops.append(crop)
            greyed.append(np.where(mask[y0:y1, x0:x1, None], crop, np.uint8(128)))
        inputs = self.processor(images=crops + greyed, input_data_format="channels_last", return_tensors="pt")
        emb = F.normalize(self.model.get_image_features(**inputs.to(self.device)).pooler_output.float(), dim=1)
        emb = F.normalize(emb[: len(crops)] + emb[len(crops) :], dim=1)
        emb[torch.from_numpy(empty).to(self.device)] = 0
        return emb

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """(n, D) L2-normalized float32 embeddings of the texts, each put in TEMPLATE and lowercased."""
        prompts = [TEMPLATE.format(t).lower() for t in texts]
        inputs = self.processor(
            text=prompts, padding="max_length", max_length=TEXT_LENGTH, truncation=True, return_tensors="pt"
        ).to(self.device)
        return F.normalize(self.model.get_text_features(**inputs).pooler_output.float(), dim=1)


@cache
def load_encoder() -> SiglipEncoder:
    """Load SigLIP 2 onto the GPU once per process; later calls return the same encoder."""
    return SiglipEncoder(MODEL_ID)


def features(run: Run, masks: Masks, encoder: SiglipEncoder) -> TextAligned:
    """The siglip stage: a SigLIP embedding for every mask of the run, row-aligned with masks."""
    boxes = mask_boxes(masks.rle)
    siglip = None
    for frame in np.unique(masks.frame):
        rows = np.flatnonzero(masks.frame == frame)
        stack = mask_utils.decode([masks.rle[r] for r in rows]).astype(bool).transpose(2, 0, 1)  # (k, H, W)
        image = footage.load_rgb(run, int(frame))
        emb = np.concatenate(
            [
                encoder.encode_masks(image, stack[i : i + BATCH], boxes[rows[i : i + BATCH]]).cpu().numpy()
                for i in range(0, len(rows), BATCH)
            ]
        )
        if siglip is None:
            siglip = np.zeros((len(masks.frame), emb.shape[1]), np.float16)
        siglip[rows] = emb.astype(np.float16)
    return TextAligned(siglip=siglip if siglip is not None else np.zeros((0, 1), np.float16))
