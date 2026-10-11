import torch
from transformers import AutoModel, AutoImageProcessor
from utils import replica

import torch.nn.functional as F
from utils.image_crop import crop_bbox

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

        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        #
        # embedding = outputs.pooler_output
        # return embedding


        with torch.inference_mode():
            outputs = self.model(**inputs)

        embedding = outputs.pooler_output
        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding

        # return embedding.squeeze(0)

    def encode_objects(self, image, bboxes, padding=0.1, batch_size=8,
                       inclusive=True):
        """
        Encode objects from an RGB image and bounding boxes.

        Args:
            image: RGB NumPy array [H, W, 3].
            bboxes: Array-like bounding boxes [N, 4] in XYXY pixel coordinates.
            padding: Relative padding around each box.
            batch_size: Maximum number of crops per DINO forward pass.
            inclusive: Whether x2/y2 are inclusive pixel coordinates.

        Returns:
            descriptors: torch.Tensor [N, D], L2-normalized.
            valid: torch.BoolTensor [N], indicating usable boxes.

        Invalid boxes produce zero descriptors.
        """
        import math
        import torch.nn.functional as F

        boxes = torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        height, width = image.shape[:2]
        descriptor_dim = self.model.config.hidden_size

        descriptors = torch.zeros(
            (len(boxes), descriptor_dim),
            device=self.device,
            dtype=torch.float32,
        )
        valid = torch.zeros(len(boxes), dtype=torch.bool, device=self.device)

        crops = []
        indices = []

        for i, box in enumerate(boxes.tolist()):
            x1, y1, x2, y2 = box

            if not all(math.isfinite(v) for v in box):
                continue

            if inclusive:
                x2 += 1
                y2 += 1

            # Reject invalid boxes before padding.
            if x2 <= x1 or y2 <= y1:
                continue

            # Reject boxes with no intersection with the image.
            if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
                continue

            crop = crop_bbox(image, (x1, y1, x2, y2), padding)

            if crop.size == 0:
                continue

            crops.append(crop.copy())
            indices.append(i)

        # Process crops in bounded batches.
        for start in range(0, len(crops), batch_size):
            batch = crops[start:start + batch_size]
            batch_indices = indices[start:start + batch_size]

            embeddings = self.encode(batch)
            embeddings = F.normalize(
                embeddings.float(), p=2, dim=-1
            )

            descriptors[batch_indices] = embeddings
            valid[batch_indices] = True

        return descriptors, valid




if __name__ == "__main__":
    from utils.image_crop import crop_bboxes

    MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    dino = DINOEncoder(MODEL_NAME)

    rgb, depth, ids = replica.load_frame("room_0", "00", 0)

    # Example SAM3 bounding boxes (inclusive endpoints)
    bboxes = [
        (100, 100, 250, 300),
        (350, 150, 500, 400),
    ]

    crops = crop_bboxes(rgb, bboxes)

    embeddings = dino.encode(crops)

    print("Number of crops:", len(crops))
    print("Embeddings shape:", embeddings.shape)
    print("Embedding norms:", torch.linalg.vector_norm(embeddings, dim=1))

    similarity = embeddings @ embeddings.T
    print("Cosine similarity matrix:")
    print(similarity)
