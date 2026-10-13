"""Depth and cameras of a run from VGGT-Omega (github.com/facebookresearch/vggt-omega), for --pose-source vggt.

VGGT-Omega reconstructs all frames of a run in one forward pass: per frame a z-depth map with its confidence and an
OpenCV camera, in the first camera's frame and in the model's own units. Nothing here uses ground truth, so it runs on
phone video. One similarity transform per run, Geometry.sim3, puts the reconstruction in metres with z up:

  up       perpendicular to every camera's x axis (right in the image), which holds when the camera isn't rolled, as
           with a handheld phone. On the 8 Replica scenes it is 0.2-0.7 degrees off. When the camera hardly turns, the
           x axes don't pin it down and the mean of the cameras' up axes is used instead.
  scale    Depth Pro's metric depth over VGGT's, the median ratio over SCALE_FRAMES frames: on Replica 5.6% off on
           average and 17.5% at worst (Depth Anything V2 metric indoor was 33% off).
  heading  along the walls: the normals of vertical surfaces (from VGGT's depth) mostly point in four directions 90
           degrees apart, and the one closest to the first camera's view becomes +x. Axis-aligned boxes then fit the
           things that stand along walls. On Replica this finds the walls' direction to within about 1 degree (Replica's
           own frame doesn't follow its walls in 6 of the 8 scenes). Without clear walls (WALL_STRENGTH), +x is the
           first camera's view direction made horizontal.

On Replica, gt_errors scores this against the GT trajectory. The checkpoints are gated: request access to
facebook/VGGT-Omega on Hugging Face (FAIR Noncommercial Research License). Depth Pro is under Apple's ML Research
license. 100 frames take about 13 GB of GPU memory and 10 s, 286 frames 28 GB and 51 s.
"""

from functools import cache

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

from data import footage, replica
from data.stages import Geometry, Run

REPO_ID = "facebook/VGGT-Omega"
CHECKPOINT = "vggt_omega_1b_512.pt"  # the authors' choice for new data; 416 is their benchmark reproduction
RESOLUTION = 512
MODE = "balanced"  # keeps about RESOLUTION² pixels: Replica's 1200 x 680 goes in as 688 x 384
DEPTH_MODEL = "apple/DepthPro-hf"
SCALE_FRAMES = 8  # frames Depth Pro sees, spread over the run
MIN_TURN = 0.2  # below this ratio of the x axes' second to first singular value, the camera hardly turns
WALL_FRAMES = 20  # frames whose depth normals find the walls
VERTICAL = 0.2  # a surface is vertical when its normal's z is below this
WALL_STRENGTH = 0.3  # 0 to 1: how much the vertical normals agree on four directions 90 degrees apart
MAX_FRAMES = 250  # 286 frames peaked at 27.9 GB of the RTX 5090's 32, after one failed allocation
PARAMS = {
    "checkpoint": CHECKPOINT,
    "resolution": RESOLUTION,
    "mode": MODE,
    "alignment": "up from camera x axes, heading from wall normals, scale from metric depth",
    "depth_model": DEPTH_MODEL,
    "scale_frames": SCALE_FRAMES,
}


@cache
def load_model() -> VGGTOmega:
    """Load VGGT-Omega onto the GPU once per process; later calls return the same model."""
    model = VGGTOmega()
    model.load_state_dict(torch.load(hf_hub_download(REPO_ID, CHECKPOINT), map_location="cpu"))
    return model.to("cuda").eval()


@cache
def load_depth_model():
    """Depth Pro's processor and model on the GPU, once per process."""
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, dtype=torch.float16).to("cuda").eval()
    return AutoImageProcessor.from_pretrained(DEPTH_MODEL), model


@torch.inference_mode()
def reconstruct(run: Run, model: VGGTOmega) -> Geometry:
    """The vggt stage: depth and cameras of every frame of the run, and the transform to metres with z up."""
    frames = footage.frame_numbers(run)
    if len(frames) > MAX_FRAMES:
        stride = -(-run.stride * len(frames) // MAX_FRAMES)
        raise ValueError(f"{len(frames)} frames, but one VGGT-Omega pass fits {MAX_FRAMES}: use --stride {stride}")
    paths = [str(footage.image_path(run, int(i))) for i in frames]
    images = load_and_preprocess_images(paths, mode=MODE, image_resolution=RESOLUTION).to("cuda")
    predictions = model(images)
    h, w = images.shape[-2:]
    height, width = footage.image_size(run)
    cam_from_world, intrinsics = encoding_to_camera(predictions["pose_enc"], (h, w))
    intrinsics = intrinsics[0].cpu().numpy()
    intrinsics[:, 0] *= width / w  # fx and cx; the aspect ratio changes a little, so x and y differ
    intrinsics[:, 1] *= height / h  # fy and cy
    geometry = Geometry(
        frame=frames,
        depth=predictions["depth"][0, ..., 0].cpu().numpy().astype(np.float16),
        conf=predictions["depth_conf"][0].cpu().numpy().astype(np.float16),
        intrinsics=intrinsics.astype(np.float32),
        cam_from_world=cam_from_world[0].cpu().numpy().astype(np.float32),
        sim3=np.eye(4, dtype=np.float32),
    )
    del predictions, images
    torch.cuda.empty_cache()
    geometry.sim3 = alignment(run, geometry).astype(np.float32)
    return geometry


def alignment(run: Run, geometry: Geometry) -> np.ndarray:
    """(4, 4) [s R | 0] from the model's world frame to metres, z up and x along the walls (module docstring)."""
    up = gravity_up(geometry.cam_from_world)
    x = geometry.cam_from_world[0, 2, :3]  # the first camera's view direction (its z axis) in world coordinates
    if abs(x @ up) > 0.95:  # looking straight up or down: use its x axis
        x = geometry.cam_from_world[0, 0, :3]
    x = x - (x @ up) * up
    x /= np.linalg.norm(x)
    rotation = np.stack([x, np.cross(up, x), up])  # model world to z up, x along the first view
    views = []
    for k in np.unique(np.linspace(0, len(geometry.frame) - 1, WALL_FRAMES).astype(int)):
        K = geometry.intrinsics[k].astype(np.float64)
        K[:2] *= (np.array(geometry.depth.shape[1:]) / footage.image_size(run))[::-1, None]  # to the depth's size
        views.append((geometry.depth[k].astype(np.float32), K, rotation @ geometry.cam_from_world[k, :, :3].T))
    angle, strength = wall_angle(views)
    if strength >= WALL_STRENGTH:
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ rotation  # turn the walls onto x and y
    sim3 = np.eye(4)
    sim3[:3, :3] = metric_scale(run, geometry) * rotation
    return sim3


def wall_angle(views) -> tuple[float, float]:
    """The direction of the walls about z, in radians modulo 90 degrees, and how clear it is (0 to 1).

    views are (depth (h, w), K for that size, (3, 3) camera-to-world rotation of a z-up world). Every pixel's normal
    comes from its neighbours' 3D points. The normals of vertical surfaces vote with 4 times their angle, so that the
    four directions of a rectangular room add up, and the mean vote's angle over 4 is the walls' direction.
    """
    total, count = 0j, 0
    for depth, K, world_from_camera in views:
        h, w = depth.shape
        v, u = np.mgrid[0:h, 0:w]
        points = np.stack([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth], axis=-1)
        normals = np.cross(points[1:-1, 2:] - points[1:-1, :-2], points[2:, 1:-1] - points[:-2, 1:-1])
        jump = (maximum_filter(depth, 3) - minimum_filter(depth, 3)) / np.maximum(depth, 1e-6)
        normals = normals[(depth[1:-1, 1:-1] > 0) & (jump[1:-1, 1:-1] < 0.05)]  # not across depth edges
        normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9) @ world_from_camera.T
        vertical = normals[np.abs(normals[:, 2]) < VERTICAL]
        total += np.exp(4j * np.arctan2(vertical[:, 1], vertical[:, 0])).sum()
        count += len(vertical)
    return float(np.angle(total) / 4), float(abs(total) / max(count, 1))


def gravity_up(cam_from_world: np.ndarray) -> np.ndarray:
    """Unit up direction in the model's world frame, from the cameras' axes (see the module docstring)."""
    x_axes, y_axes = cam_from_world[:, 0, :3], cam_from_world[:, 1, :3]  # right and down in the image
    _, singular, vt = np.linalg.svd(x_axes)
    up = -y_axes.mean(0)
    if singular[1] >= MIN_TURN * singular[0]:
        up = vt[2] * np.sign(vt[2] @ up)
    return up / np.linalg.norm(up)


@torch.inference_mode()
def metric_scale(run: Run, geometry: Geometry) -> float:
    """Metres per model unit: the median, over SCALE_FRAMES frames, of Depth Pro's depth over VGGT's."""
    processor, model = load_depth_model()
    size = footage.image_size(run)
    ratios = []
    for k in np.unique(np.linspace(0, len(geometry.frame) - 1, SCALE_FRAMES).astype(int)):
        inputs = processor(images=footage.load_rgb(run, int(geometry.frame[k])), return_tensors="pt")
        outputs = model(**inputs.to("cuda", torch.float16))
        metric = processor.post_process_depth_estimation(outputs, target_sizes=[size])[0]["predicted_depth"]
        metric = metric.float().cpu().numpy()
        depth = upsample(geometry.depth[k], size)
        valid = (depth > 0) & (metric > 0) & np.isfinite(metric)
        ratios.append(np.median(metric[valid] / depth[valid]))
    return float(np.median(ratios))


def upsample(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A depth map at the model's resolution, bilinearly resized to (height, width), float32."""
    small = Image.fromarray(depth.astype(np.float32))
    return np.asarray(small.resize((size[1], size[0]), Image.Resampling.BILINEAR))


def gt_errors(run: Run, geometry: Geometry) -> dict[str, float]:
    """For Replica runs, how far the reconstruction and alignment are from the GT trajectory (scoring only).

    ate_m is the RMS camera position error after the best-fit Sim(3) from the predicted to the GT camera centres,
    scale_error the alignment's scale relative to that fit's, up_error_deg the angle between the estimated and the GT
    up direction (Replica's z), and wall_error_deg how far the aligned x axis is from the walls (modulo 90 degrees),
    whose direction comes from the same normals computed on the GT depth and poses.
    """
    scene, seq = replica.parse_source(run.source)
    gt = replica.load_poses(scene, seq)[geometry.frame, :3, 3]
    centres = camera_centres(geometry.cam_from_world)
    fit = umeyama(centres, gt)
    fit_scale = float(np.linalg.norm(fit[:3, 0]))
    up_true = fit[:3, :3].T @ np.array([0.0, 0.0, 1.0]) / fit_scale
    up_est = geometry.sim3[2, :3] / geometry.scale
    # The aligned frame's x axis in the GT frame, against the GT walls' direction
    to_gt = fit[:3, :3] / fit_scale @ (geometry.sim3[:3, :3] / geometry.scale).T
    heading = np.arctan2(to_gt[1, 0], to_gt[0, 0])
    views = []
    for i in np.linspace(0, replica.N_FRAMES - 1, WALL_FRAMES).astype(int):
        K = replica.K.copy()
        K[:2] /= 2
        views.append((replica.load_depth(scene, seq, int(i))[::2, ::2], K, replica.load_poses(scene, seq)[i, :3, :3]))
    walls, _ = wall_angle(views)
    wall_error = np.degrees((walls - heading + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    return {
        "ate_m": float(np.sqrt(((centres @ fit[:3, :3].T + fit[:3, 3] - gt) ** 2).sum(1).mean())),
        "scale_error": geometry.scale / fit_scale - 1,
        "up_error_deg": float(np.degrees(np.arccos(np.clip(up_true @ up_est, -1, 1)))),
        "wall_error_deg": float(abs(wall_error)),
    }


def camera_centres(cam_from_world: np.ndarray) -> np.ndarray:
    """(F, 3) camera positions in the world frame of (F, 3, 4) camera-from-world matrices."""
    rotation, translation = cam_from_world[:, :, :3], cam_from_world[:, :, 3]
    return -np.einsum("fji,fj->fi", rotation, translation)


def umeyama(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(4, 4) [s R | t] that maps the (n, 3) source points closest to target in least squares (Umeyama, 1991)."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    xs, xt = source - mu_s, target - mu_t
    u, d, vt = np.linalg.svd(xt.T @ xs / len(source))
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    rotation = u @ flip @ vt
    scale = np.trace(np.diag(d) @ flip) / (xs**2).sum(1).mean()
    sim3 = np.eye(4)
    sim3[:3, :3] = scale * rotation
    sim3[:3, 3] = mu_t - scale * rotation @ mu_s
    return sim3


def world_from_camera(geometry: Geometry, k: int) -> np.ndarray:
    """(4, 4) camera-to-world matrix of row k in the aligned world frame (metres, z up), for depth in metres.

    A camera point x in model units lands at sim3 @ (c2w @ x). With depth in metres (s x), the rotation is R R_c and the
    translation s R t_c + t, where [R_c | t_c] is the model's camera-to-world and sim3 = [s R | t].
    """
    s = geometry.scale
    rotation, translation = geometry.sim3[:3, :3] / s, geometry.sim3[:3, 3]
    r_cw, t_cw = geometry.cam_from_world[k, :, :3], geometry.cam_from_world[k, :, 3]
    c2w = np.eye(4)
    c2w[:3, :3] = rotation @ r_cw.T
    c2w[:3, 3] = s * rotation @ (-r_cw.T @ t_cw) + translation
    return c2w
