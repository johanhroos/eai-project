"""The frames of a run, whatever its source: a Replica sequence or a phone video.

  replica/<scene>/<seq>  vMAP's Replica renders (data/replica.py), 1200 x 680 PNGs
  phone/<name>           a video at data/phone/<name>.<ext>, any container ffmpeg reads. The frames stage (extract)
                         decodes every stride-th frame, upright as a player shows it, and saves it as a JPEG with the
                         long side at most MAX_SIDE pixels.
  phone/<a>+<b>          several videos of the same place as one run: their frames are numbered one after the other.
                         SAM tracks each video on its own (sam.segment_clips), VGGT-Omega reconstructs all of them in
                         one pass, so they share one world frame, and the graph links their objects.

Every stage reads frames through this module, so it runs on either source.
"""

import numpy as np
from PIL import Image
from torchcodec.decoders import VideoDecoder

from data import replica
from data.paths import DATA_DIR
from data.stages import Frames, Run, load_stage, save_stage

PHONE_DIR = DATA_DIR / "phone"
MAX_SIDE = 1280  # Replica's frames are 1200 wide; phone video is often 1920 or 3840
JPEG_QUALITY = 95
HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}  # HLG and PQ, as phones record HDR


def is_replica(run: Run) -> bool:
    return run.source.startswith("replica/")


def make_run(source: str, stride: int, mask_source: str, pose_source: str | None, vocab: str = "") -> Run:
    """The Run of command-line arguments. pose_source None is Replica's poses on Replica and VGGT-Omega otherwise.

    Raises ValueError for a phone video with GT masks or poses, which it doesn't have.
    """
    replica_source = source.startswith("replica/")
    pose_source = pose_source or ("gt" if replica_source else "vggt")
    if not replica_source and (mask_source, pose_source) != ("sam3", "vggt"):
        raise ValueError(f"{source} has no ground truth: use --mask-source sam3 --pose-source vggt")
    return Run(source, stride, mask_source, pose_source, vocab if mask_source == "sam3" else "")


def video_paths(run: Run) -> list:
    """The video files of a phone run, in order: for each name in phone/<a>+<b>, the one file data/phone/<name>.* ."""
    if not run.source.startswith("phone/"):
        raise ValueError(f"{run.source!r} is not a phone run")
    paths = []
    for name in run.source.removeprefix("phone/").split("+"):
        found = sorted(PHONE_DIR.glob(f"{name}.*"))
        if len(found) != 1:
            raise FileNotFoundError(f"{run.source!r}: expected one video file data/phone/{name}.<ext>, found {found}")
        paths.append(found[0])
    return paths


def frame_numbers(run: Run) -> np.ndarray:
    """(F,) int32 frame numbers of the run, in temporal order."""
    if is_replica(run):
        return np.array(replica.frame_numbers(run.stride), np.int32)
    return load_stage(run.stage_dir("frames"), Frames).frame


def clip_index(run: Run) -> np.ndarray:
    """(F,) int16: which of the run's videos each frame (frame_numbers) is from. All 0 for Replica and single videos."""
    if is_replica(run):
        return np.zeros(len(frame_numbers(run)), np.int16)
    frames = load_stage(run.stage_dir("frames"), Frames)
    return frames.clip if frames.clip is not None else np.zeros(len(frames.frame), np.int16)


def clip_names(run: Run) -> list[str]:
    """Names of the run's videos, in the order clip_index counts them; the source itself on Replica."""
    if is_replica(run):
        return [run.source]
    return load_stage(run.stage_dir("frames"), Frames).clips or [run.source.removeprefix("phone/")]


def frame_label(run: Run, i: int) -> str:
    """Frame i as "<video> frame <n>", n counting within that video; "<source> frame <i>" on Replica."""
    if is_replica(run):
        return f"{run.source} frame {i}"
    numbers, clip = frame_numbers(run), clip_index(run)
    k = int(clip[numbers == i][0])
    return f"{clip_names(run)[k]} frame {i - int(numbers[clip == k][0])}"


def image_path(run: Run, i: int):
    if is_replica(run):
        scene, seq = replica.parse_source(run.source)
        return replica.sequence_dir(scene, seq) / "rgb" / f"rgb_{i}.png"
    return run.stage_dir("frames") / f"{i:06d}.jpg"


def load_rgb(run: Run, i: int) -> np.ndarray:
    """Frame i as RGB (H, W, 3) uint8, writable: torch warns about read-only arrays (Depth Pro's processor in vggt)."""
    with Image.open(image_path(run, i)) as image:
        return np.array(image.convert("RGB"))


def load_run_rgb(run: Run) -> tuple[list[np.ndarray], list[int]]:
    """All frames of the run in temporal order and their frame numbers. Stride 1 on Replica is about 5 GB."""
    numbers = frame_numbers(run).tolist()
    return [load_rgb(run, i) for i in numbers], numbers


def image_size(run: Run) -> tuple[int, int]:
    """(height, width) of the run's frames."""
    if is_replica(run):
        return replica.HEIGHT, replica.WIDTH
    frames = load_stage(run.stage_dir("frames"), Frames)
    return frames.height, frames.width


def extract(run: Run, batch=16) -> None:
    """The frames stage of a phone run: decode every stride-th frame of its videos and save each as a JPEG.

    torchcodec applies the videos' rotation metadata, so portrait clips come out upright, as ffmpeg and players show
    them. Frames are only ever downscaled. The videos of one run must come out the same size: film them the same way up.
    """
    out = run.stage_dir("frames")
    out.mkdir(parents=True, exist_ok=True)
    numbers, clips, sizes, fps, first = [], [], set(), [], 0
    for k, path in enumerate(video_paths(run)):
        decoder = VideoDecoder(str(path))
        if decoder.metadata.color_transfer_characteristic in HDR_TRANSFERS:
            print(
                f"Warning: {path.name} is HDR, and its frames are decoded without tone mapping, so colours look flat. "
                "Record in SDR (on an iPhone, turn off Settings > Camera > Record Video > HDR Video) if you can."
            )
        local = np.arange(0, decoder.metadata.num_frames, run.stride)
        for start in range(0, len(local), batch):
            chunk = local[start : start + batch].tolist()
            for i, frame in zip(chunk, decoder.get_frames_at(chunk).data, strict=True):  # (C, H, W) uint8
                image = Image.fromarray(frame.permute(1, 2, 0).numpy())
                scale = min(1.0, MAX_SIDE / max(image.size))
                if scale < 1:
                    size = round(image.width * scale), round(image.height * scale)
                    image = image.resize(size, Image.Resampling.LANCZOS)
                image.save(out / f"{first + i:06d}.jpg", quality=JPEG_QUALITY)
                sizes.add((image.height, image.width))
        numbers.append(first + local)
        clips.append(np.full(len(local), k))
        first += decoder.metadata.num_frames
        fps.append(float(decoder.metadata.average_fps))
    if len(sizes) > 1:
        raise ValueError(f"{run.source}: the videos' frames differ in size {sorted(sizes)}; film them the same way up")
    (height, width), names = sizes.pop(), [p.stem for p in video_paths(run)]
    numbers, clips = np.concatenate(numbers).astype(np.int32), np.concatenate(clips).astype(np.int16)
    frames = Frames(numbers, height, width, fps[0], clips, names)
    save_stage(out, frames, {"videos": [str(p) for p in video_paths(run)], "max_side": MAX_SIDE})
