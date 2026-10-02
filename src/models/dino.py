import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "facebook/dinov3-vitl16-pretrain-lvd1689m"
)

model.eval()
