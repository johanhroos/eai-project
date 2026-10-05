import torch
from transformers import AutoModel, AutoImageProcessor
from utils import replica



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


if __name__ == "__main__":
    MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    dino = DINOEncoder(MODEL_NAME)

    print(next(dino.model.parameters()).device)

    # load image
    rgb, depth, ids = replica.load_frame("room_0", "00", 0)
    print(rgb.shape)

    embedding = dino.encode(rgb.copy())
    print(embedding.shape)
