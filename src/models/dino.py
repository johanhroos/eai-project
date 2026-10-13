from functools import cache

import numpy as np
import pycocotools.mask as mask_utils
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from data import footage
from data.stages import Features, Masks, Run
from models.crops import pad_bbox
from models.sam import mask_boxes

# Usage:
# 1. Create a DINOEncoder with the desired DINO model.
# 2. Load an RGB image as a NumPy array.
# 3. Pass the image to encode().
# 4. encode() returns a 1D DINO feature embedding for the image.
#
# Example:
#   dino = DINOEncoder("facebook/dinov3-vitl16-pretrain-lvd1689m")
#   embedding = dino.encode(rgb)
#   print(embedding.shape)
#
# encode_masks() gives one embedding per object mask instead, and features() runs it over a whole run.

MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
BATCH = 32  # masks encoded at once: a frame with all its masks at once can overflow a 10 GB GPU


class DINOEncoder:
    def __init__(self, model_name, device="cuda"):
        self.device = torch.device(device)

        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def encode(self, image):
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.pooler_output

        return embedding.squeeze(0)

    @torch.inference_mode()
    def encode_masks(self, image: np.ndarray, masks: np.ndarray, boxes: np.ndarray, padding=0.1) -> torch.Tensor:
        """One L2-normalized embedding per mask, (k, D) float32 on the device.

        image is RGB (H, W, 3) uint8, masks (k, H, W) bool and boxes their (k, 4) x0, y0, x1, y1 from mask_boxes.
        Each mask's padded box is cropped and encoded, and the patch tokens are averaged with the share of each patch
        the mask covers as weight, so the background in the crop counts little. An empty mask gets a zero vector.
        """
        crops, weights = [], []
        side = self.model.config.image_size // self.model.config.patch_size  # the processor resizes to a square
        for mask, box in zip(masks, boxes, strict=True):
            if box[2] <= box[0] or box[3] <= box[1]:  # empty mask: encode anything, its weights are zero
                x0, y0, x1, y1 = 0, 0, 1, 1
            else:
                x0, y0, x1, y1 = pad_bbox(box, image.shape, padding)
            crops.append(image[y0:y1, x0:x1].copy())  # torch warns about the read-only arrays PIL gives
            cover = torch.from_numpy(mask[y0:y1, x0:x1]).float()[None]
            weights.append(F.adaptive_avg_pool2d(cover, side).flatten())
        inputs = self.processor(images=crops, input_data_format="channels_last", return_tensors="pt").to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = self.model(**inputs).last_hidden_state.float()  # (k, 1 + registers + side², D), layer-normed
        patches = tokens[:, 1 + self.model.config.num_register_tokens :]
        w = torch.stack(weights).to(self.device)
        pooled = (w[:, :, None] * patches).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-6)
        return F.normalize(pooled, dim=1)


@cache
def load_encoder() -> DINOEncoder:
    """Load DINOv3 onto the GPU once per process; later calls return the same encoder."""
    return DINOEncoder(MODEL_ID)


def features(run: Run, masks: Masks, encoder: DINOEncoder) -> Features:
    """The features stage: a DINO embedding for every mask of the run, row-aligned with masks."""
    boxes = mask_boxes(masks.rle)
    dino = np.zeros((len(masks.frame), encoder.model.config.hidden_size), np.float16)
    for frame in np.unique(masks.frame):
        rows = np.flatnonzero(masks.frame == frame)
        stack = mask_utils.decode([masks.rle[r] for r in rows]).astype(bool).transpose(2, 0, 1)  # (k, H, W)
        image = footage.load_rgb(run, int(frame))
        for i in range(0, len(rows), BATCH):
            embeddings = encoder.encode_masks(image, stack[i : i + BATCH], boxes[rows[i : i + BATCH]])
            dino[rows[i : i + BATCH]] = embeddings.cpu().numpy().astype(np.float16)
    return Features(dino=dino)
