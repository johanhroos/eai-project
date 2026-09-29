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

## Data

```bash
bash scripts/download_data.sh   # 45 GB download; needs wget, unzip and ~95 GB free while running (46 GB after)
```

Downloads the 8 Replica scenes that have OpenLex3D labels (room0–2, office0–4) into `data/`. Set `DATA_DIR` to put them
elsewhere. Rerunning resumes an interrupted download.

| Path | Source | Contents |
|---|---|---|
| `data/vmap/<scene>/imap/{00,01}/` | [vMAP](https://github.com/kxhit/vMAP) | Two camera paths × 2000 frames: `rgb/`, `depth/` (uint16, mm), `semantic_instance/` (Replica object ID per pixel), `traj_w_c.txt` (camera-to-world 4×4 per line). Path `00` is the NICE-SLAM trajectory. |
| `data/vmap/<scene>/habitat/info_semantic.json` | vMAP | Object ID → Replica class name |
| `data/gt_openlex3d/replica/<scene>/` | [OpenLex3D](https://github.com/openlex3d/openlex3d) | Open-vocabulary labels and text queries per object ID |

Camera: 1200×680, fx = fy = 600, cx = 599.5, cy = 339.5. Frames are numbered `0`–`1999` without zero padding, so sort
by number. Scene names are `room_0` in vMAP and `room0` in OpenLex3D.

## SAM 3 preview

Request access to [facebook/sam3](https://huggingface.co/facebook/sam3) and authenticate on the GPU machine
with `huggingface_hub.login()` once. Run the **SAM 3 short clip** section of `notebooks/explore_data.ipynb`
in VS Code with the project's `.venv` kernel, or generate the same cache from the terminal:

```bash
uv run --no-sync python -m models.sam3 --scene room_0 --frames 60 --prompt lamp
```

The video tracker saves original-resolution masks, SAM track IDs, scores and boxes under `outputs/sam3_preview/`.
The notebook displays RGB and predicted masks side by side; set `RUN_SAM = False` to inspect the existing cache.
Reusing the output directory overwrites that run. Track IDs belong to the clip, not the persistent object graph.
The optional `kernels` package is not included; Transformers warns and skips NMS and mask cleanup without it.
