"""Load the Replica frames (vMAP) and object labels (OpenLex3D) that scripts/download_data.sh puts in DATA_DIR.

Poses are camera-to-world with OpenCV camera axes (x right, y down, z forward), and the world frame is z-up.
Scenes use the vMAP names: room_0..2, office_0..4. Sequences are "00" (the NICE-SLAM camera path) and "01".
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from data.paths import DATA_DIR
from data.stages import Masks, Run, encode_rle

SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]
N_FRAMES = 2000
WIDTH, HEIGHT = 1200, 680
K = np.array([[600.0, 0.0, 599.5], [0.0, 600.0, 339.5], [0.0, 0.0, 1.0]])
# Replica classes that name no kind of object (unlabelled, catch-all or blurred-out), so make no sense as a text prompt
NON_OBJECTS = {"undefined", "non-plane", "anonymize_picture", "anonymize_text"}
# Left out of the evaluation and the views: nothing to re-identify, or never prompted
IGNORE_CLASSES = NON_OBJECTS | {"wall", "floor", "ceiling"}


@dataclass
class ReplicaObject:
    id: int
    class_name: str  # Replica class
    # OpenLex3D labels by category, empty for objects OpenLex3D leaves out. synonyms, vis_sim and depictions hold
    # names; clutter holds the IDs (as strings) of nearby objects, not names.
    labels: dict[str, list[str]] = field(default_factory=dict)


def sequence_dir(scene: str, seq: str = "00") -> Path:
    return DATA_DIR / "vmap" / scene / "imap" / seq


def load_poses(scene: str, seq: str = "00") -> np.ndarray:
    """Camera-to-world 4x4 matrices, one per frame: shape (N_FRAMES, 4, 4)."""
    return np.loadtxt(sequence_dir(scene, seq) / "traj_w_c.txt").reshape(-1, 4, 4)


def frame_numbers(stride: int) -> range:
    """The frames a run with this stride uses: every stride-th frame from 0."""
    return range(0, N_FRAMES, stride)


def load_rgb(scene: str, seq: str, i: int) -> np.ndarray:
    """Frame i as RGB (H, W, 3) uint8."""
    return _load_png(sequence_dir(scene, seq) / "rgb" / f"rgb_{i}.png")


def load_depth(scene: str, seq: str, i: int) -> np.ndarray:
    """Frame i as depth (H, W) float32 in metres, 0 where there is no depth."""
    return _load_png(sequence_dir(scene, seq) / "depth" / f"depth_{i}.png").astype(np.float32) / 1000


def load_instance_ids(scene: str, seq: str, i: int) -> np.ndarray:
    """Frame i as Replica object IDs (H, W) uint16, 0 where there is no object."""
    return _load_png(sequence_dir(scene, seq) / "semantic_instance" / f"semantic_instance_{i}.png")


def load_objects(scene: str) -> dict[int, ReplicaObject]:
    """All objects in the scene by ID, the value the instance masks hold (0 is no object)."""
    with (DATA_DIR / "vmap" / scene / "habitat" / "info_semantic.json").open() as f:
        objects = {o["id"]: ReplicaObject(o["id"], o["class_name"]) for o in json.load(f)["objects"]}
    with (DATA_DIR / "gt_openlex3d" / "replica" / scene.replace("_", "") / "gt_categories.json").open() as f:
        for sample in json.load(f)["dataset"]["samples"]:
            objects[sample["object_id"]].labels = sample["labels"]["image_attributes"]
    return objects


def load_queries(scene: str, level: str = "level0") -> dict[str, list[int]]:
    """OpenLex3D text queries of the scene and the object IDs each one describes.

    level0 queries name an object (its synonyms, such as "throw pillow"); level1 add a pattern, material or picture
    seen on it, such as "stripe pillow" or "wood table".
    """
    scene_dir = DATA_DIR / "gt_openlex3d" / "replica" / scene.replace("_", "")
    with (scene_dir / "gt_categories_query_to_object_mapping_all.json").open() as f:
        return json.load(f)[level]


def class_names(objects: dict[int, ReplicaObject], promptable: bool = False) -> list[str]:
    """The distinct Replica classes of these objects, sorted. promptable leaves out NON_OBJECTS."""
    names = {o.class_name for o in objects.values()}
    return sorted(names - NON_OBJECTS if promptable else names)


def parse_source(source: str) -> tuple[str, str]:
    """Scene and sequence of a Run.source such as "replica/room_0/00"."""
    dataset, scene, seq = source.split("/")
    if dataset != "replica" or scene not in SCENES or seq not in ("00", "01"):
        raise ValueError(f"not a Replica source: {source!r}")
    return scene, seq


def prompt_vocab(run: Run) -> list[str]:
    """The scene's Replica classes that make sense as text prompts.

    gt_masks keeps every class, so a vocab from this differs from the GT one: match them by name, not by index.
    """
    scene, _ = parse_source(run.source)
    return class_names(load_objects(scene), promptable=True)


def gt_masks(run: Run) -> tuple[Masks, list[str]]:
    """Ground-truth instance masks for every stride-th frame, in the same format SAM 3 produces, plus their vocab.

    One row per object visible in a frame. sam_id is the Replica object ID, prompt indexes the vocab (the scene's
    Replica class names, sorted) and score is 1. Pixels whose ID is missing from info_semantic.json are skipped.
    """
    scene, seq = parse_source(run.source)
    objects = load_objects(scene)
    vocab = class_names(objects)
    prompt_of = {o.id: vocab.index(o.class_name) for o in objects.values()}

    frame, prompt, sam_id, rle = [], [], [], []
    for i in frame_numbers(run.stride):
        ids = load_instance_ids(scene, seq, i)
        visible = [j for j in np.unique(ids).tolist() if j in prompt_of]
        if not visible:
            continue
        rle += encode_rle(ids == np.array(visible, dtype=ids.dtype)[:, None, None])
        frame += [i] * len(visible)
        prompt += [prompt_of[j] for j in visible]
        sam_id += visible

    masks = Masks(
        frame=np.array(frame, dtype=np.int32),
        prompt=np.array(prompt, dtype=np.int16),
        sam_id=np.array(sam_id, dtype=np.int32),
        score=np.ones(len(frame), dtype=np.float32),
        rle=rle,
    )
    return masks, vocab


def _load_png(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im)
