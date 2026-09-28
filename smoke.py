import torch
import torch.nn.functional as F
from torchcodec.decoders import VideoDecoder

print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())
x = torch.randn(4, 8, 1024, 64, device="cuda", dtype=torch.bfloat16)
print("sdpa", F.scaled_dot_product_attention(x, x, x).shape)
f = torch.compile(lambda a: (a @ a.transpose(-1, -2)).softmax(-1))
print("compile", f(x).shape)
print("decode", VideoDecoder("sample.mp4")[0].shape)  # add device="cuda" if you decode on GPU
print("free/total GB", [round(v / 1e9, 1) for v in torch.cuda.mem_get_info()])
