import sys

print("=== Python ===")
print(sys.version)

print("\n=== PyTorch ===")
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
        "GiB",
    )

print("\n=== CUDA computation ===")

if torch.cuda.is_available():
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    print("Matrix multiplication:", y.shape)
    print("Device:", y.device)
    print("CUDA computation: OK")
else:
    print("CUDA computation: SKIPPED")

print("\n=== Packages ===")

packages = [
    "numpy",
    "PIL",
    "transformers",
]

for package in packages:
    try:
        module = __import__(package)
        version = getattr(module, "__version__", "unknown")
        print(f"{package}: OK ({version})")
    except Exception as e:
        print(f"{package}: FAILED ({e})")

print("\n=== Result ===")
print("Smoke test finished.")