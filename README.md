# eai-project

DD2600 course project.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and FFmpeg (used by torchcodec for video decoding).

```bash
git clone https://github.com/johanhroos/eai-project.git
cd eai-project
uv sync --extra cu126   # or for blackwell: uv sync --extra cu130
```

Pick the extra that matches the GPU:

| Extra   | GPU                                  | Notes                                             |
|---------|--------------------------------------|---------------------------------------------------|
| `cu130` | RTX 50-series (e.g. RTX 5090)        | Blackwell needs this; cu126 has no sm_120 kernels |
| `cu126` | H100                                 | Also use it if `nvidia-smi` shows a driver < 580  |

## Check the environment

```bash
uv run --no-sync python3 -m smoke
```

It prints the torch/CUDA versions and GPU, then runs SDPA, `torch.compile`, and a video decode.

IMPORTANT: Use `uv run --no-sync` for everything: a plain `uv run` re-syncs without the extra and replaces the CUDA build of
torch with the default one from PyPI.
