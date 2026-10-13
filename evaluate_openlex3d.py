"""Score object maps with OpenLex3D's own evaluation code, to compare them with the methods in its paper.

OpenLex3D (github.com/openlex3d/openlex3d, Kassab et al., NeurIPS 2025) scores ConceptGraphs, HOV-SG, OpenMask3D and
Kassab2024 on the same 8 Replica scenes, and publishes their predictions.

evaluation/metrics.py scores whole objects against the GT masks of the frames; OpenLex3D scores points against
Replica's mesh.
A run's map is exported in OpenLex3D's prediction format, one directory per scene, PREDICTIONS/<method>/replica/<scene>:

  point_cloud.pcd  each object as a node is: its box with its points, those its views agree on that lie inside the
                   box (graph.multiview.object_points, inside_boxes), at most one per VOXEL cube, in Replica's mesh
                   frame
  index.npy        (n_points,) int64, the row of embeddings.npy of each point
  embeddings.npy   (n_objects, D) float32, each object's SigLIP 2 embedding: the mean over its masks, as graph.query
                   searches them (graph.query.object_embeddings)

Objects without a reliable mask have no points and are left out, and so are the masks of dropped nodes (-1). With
--pose-source vggt the map is first moved into Replica's frame by the Sim(3) that best maps VGGT-Omega's camera centres
onto the GT ones, the usual alignment for scoring a monocular reconstruction. The scale is fitted too, so a scale error
of the reconstruction does not count; the alignment's camera error (ATE) is saved with the scores.

The two tasks run as OpenLex3D's scripts evaluate_segmentation.py and evaluate_queries.py run them, calling their
functions one by one with the text encoder registered here:

  segmentation  every point of the GT cloud of what the cameras saw takes the top-n of the dataset's 1154 prompts, by
                cosine with the embedding of the nearest predicted point (after thinning the predicted cloud to 5 cm),
                or is missing when that point is over 5 cm away. Per GT object, the share of its points whose top-n
                holds a synonym, else a depiction, a visually similar label, clutter, else none (incorrect), or that
                are missing; averaged over the objects of a scene and then over the scenes. Walls, floors and ceilings
                are left out. --set-ranking adds the set ranking scores (slow).
  queries       each OpenLex3D query ranks the objects by cosine, and its TOP_K best are its predictions: the GT mesh
                vertices within 3.2 cm of each one's points (faiss's squared distance under 0.001). AP is scored as
                for ScanNet's instance segmentation, pooled over the scenes: mAP over IoU 0.5 to 0.95, AP50 and AP25,
                in percent. l0 queries name an object ("throw pillow"), l1 add a pattern, material or picture on it
                ("stripe pillow"), all is both.

Our maps are read with SigLIP 2's text encoder (models/siglip.py, with its prompt template), the paper's predictions
with CLIP ViT-H-14 (laion2b_s32b_b79k), as in the paper. The scores of every method go to RESULTS/<method>.json, and
the tables list every method scored so far.

Install the evaluation code once (the cu130 extra as for the rest, see the README), and get the paper's predictions:

    uv sync --extra cu130 --extra openlex3d
    wget http://aisdatasets.cs.uni-freiburg.de/openlex3d/predictions_sparse.zip  # 2 GB, unzip into BASELINES' parent

    uv run --no-sync python evaluate_openlex3d.py                                  # our graph, SAM 3, GT poses
    uv run --no-sync python evaluate_openlex3d.py --pose-source vggt --variants graph memory tracks
    uv run --no-sync python evaluate_openlex3d.py --mask-source gt                 # a map from the GT masks
    uv run --no-sync python evaluate_openlex3d.py --baselines concept-graphs   # should give the paper's Table 4 row
"""

import argparse
import json
import sys
import tempfile
import types
from functools import cache
from pathlib import Path

import numpy as np

from data import footage, replica, vocab
from data.paths import DATA_DIR, OUTPUT_DIR
from data.stages import Boxes, Graph, Masks, Run, TextAligned, load_stage

PREDICTIONS = OUTPUT_DIR / "openlex3d" / "predictions"  # PREDICTIONS/<method>/replica/<scene>, as OpenLex3D expects
RESULTS = OUTPUT_DIR / "openlex3d" / "results"
BASELINES = DATA_DIR / "openlex3d_predictions" / "extracted" / "predictions_sparse"
# Metres: an exported object keeps one point per cube. ConceptGraphs' points are 7 mm apart, HOV-SG's 19 mm. On room0,
# 1 and 2 cm score the same (all-query mAP 15.1, AP25 34.6), and 2 cm has a quarter of the points to align
VOXEL = 0.02
VARIANTS = {
    "graph": "the graph stage's nodes",
    "memory": "the object memory's nodes before graph/refine.py",
    "tracks": "the raw SAM tracks",
}
LEVELS = ("l0", "l1", "all")
TOP_K = 10  # predictions per query, OpenLex3D's default
SIGLIP = "siglip2"  # the name the text encoder is registered under, as a module of openlex3d.models
AP_KEYS = {"all_ap": "mAP", "all_ap_50%": "AP50", "all_ap_25%": "AP25"}
# The paper's Table 4: Replica, all queries pooled over the 8 scenes, AP in percent. Scoring ConceptGraphs' published
# predictions here gives the same to 0.01 (5.86, 11.32, 22.40), so the other methods are taken from the paper. Its
# OpenMask3D row adds non-maximum suppression (IoU 0.5) to the predictions it publishes
PAPER_QUERIES = {
    "ConceptGraphs": (5.86, 11.32, 22.39),
    "ConceptGraphs (GPT)": (5.13, 10.77, 18.19),
    "HOV-SG": (5.76, 11.67, 25.30),
    "Kassab2024": (1.38, 2.87, 7.54),
    "OpenMask3D + NMS": (11.47, 17.01, 24.02),
}
CATEGORY_COLUMNS = {
    "synonyms": "synonym",
    "depictions": "depict",
    "vis_sim": "vis sim",
    "clutter": "clutter",
    "incorrect": "wrong",
    "missing": "missing",
}


def scene_name(scene: str) -> str:
    """OpenLex3D's name of a Replica scene: room0 for vMAP's room_0."""
    return scene.replace("_", "")


def method_name(run: Run, variant: str) -> str:
    """Such as ours-sam3_vggt-stride20-graph, or ours-sam3_indoor_gt-stride10-memory."""
    tag = "_".join(x for x in (run.mask_source, run.vocab, run.pose_source) if x)
    return f"ours-{tag}-stride{run.stride}-{variant}"


def object_ids(run: Run, variant: str, masks: Masks) -> np.ndarray:
    """The object ID of every mask in a variant (VARIANTS), -1 for masks of no object."""
    if variant == "tracks":
        return masks.sam_id.astype(np.int32)
    if variant == "graph":
        return load_stage(run.stage_dir("graph"), Graph).node
    if variant == "memory":  # replayed with the settings the graph stage was built with
        from graph import build
        from graph.memory import AssociationConfig

        graph_metadata = json.loads((run.stage_dir("graph") / "metadata.json").read_text())
        return build.rebuild(run, AssociationConfig(**graph_metadata["config"]), save=False).graph.node
    raise ValueError(f"unknown variant {variant!r}")


def to_replica_frame(run: Run) -> tuple[np.ndarray, float]:
    """(4, 4) Sim(3) from the run's world frame to Replica's mesh frame, and the camera error (ATE, m) after it.

    The identity with Replica's poses, which are in the mesh frame. With VGGT-Omega's, the Sim(3) that best maps the
    camera centres onto the GT ones (Umeyama).
    """
    if run.pose_source == "gt":
        return np.eye(4), 0.0
    from data.stages import Geometry
    from models.vggt import umeyama, world_from_camera

    geometry = load_stage(run.stage_dir("vggt"), Geometry)
    ours = np.array([world_from_camera(geometry, k)[:3, 3] for k in range(len(geometry.frame))])
    scene, seq = replica.parse_source(run.source)
    gt = replica.load_poses(scene, seq)[geometry.frame, :3, 3]
    fit = umeyama(ours, gt)
    ate = float(np.sqrt(((ours @ fit[:3, :3].T + fit[:3, 3] - gt) ** 2).sum(1).mean()))
    return fit, ate


def inside_boxes(kept: dict[int, np.ndarray], graph: Graph | None = None) -> dict[int, np.ndarray]:
    """Each node's points (graph.multiview.object_points) inside its box, so that a node is its box with its points.

    The box is the graph stage's own where it has one (graph is that stage, whose node IDs key kept); otherwise it is
    drawn as graph.multiview.object_boxes draws it, without the outer TRIM percent of the points on each side of each
    axis.
    """
    from graph.multiview import TRIM

    clipped = {}
    for n, points in kept.items():
        if graph is not None and graph.box_min is not None:
            lo, hi = graph.box_min[n], graph.box_max[n]
        else:
            lo, hi = np.percentile(points, (TRIM, 100 - TRIM), axis=0)
        clipped[n] = points[((points >= lo) & (points <= hi)).all(axis=1)]
    return clipped


def prediction(run: Run, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """A run's map in OpenLex3D's format, plus what to know about it.

    Returns points (n, 3) float32 in Replica's frame, index (n,) int64 and embeddings (m, D) float32.
    """
    from graph import multiview
    from graph.query import object_embeddings

    masks = load_stage(run.stage_dir("masks"), Masks)
    boxes = load_stage(run.stage_dir("boxes"), Boxes)
    emb = load_stage(run.stage_dir("siglip"), TextAligned)
    ids = object_ids(run, variant, masks)
    graph = load_stage(run.stage_dir("graph"), Graph) if variant == "graph" else None
    kept = inside_boxes(multiview.object_points(run, masks, boxes, ids, max_points=None, voxel=VOXEL), graph)
    objects, object_emb = object_embeddings(masks, emb, ids)
    with_points = [n for n in objects.tolist() if len(kept.get(n, ()))]
    row = {int(o): k for k, o in enumerate(objects.tolist())}
    points = np.concatenate([kept[n] for n in with_points] + [np.zeros((0, 3), np.float32)])
    index = np.concatenate([np.full(len(kept[n]), k) for k, n in enumerate(with_points)] + [np.zeros(0)]).astype(
        np.int64
    )
    fit, ate = to_replica_frame(run)
    points = (points @ fit[:3, :3].T + fit[:3, 3]).astype(np.float32)
    embeddings = object_emb[[row[n] for n in with_points]].astype(np.float32)
    graph_metadata = run.stage_dir("graph") / "metadata.json"
    about = {
        "objects": len(with_points),
        "objects_without_points": len(objects) - len(with_points),
        "points": len(points),
        "ate_m": ate,
        "graph_algorithm": json.loads(graph_metadata.read_text()).get("algorithm") if graph_metadata.exists() else None,
    }
    return points, index, embeddings, about


def write_prediction(path: Path, points: np.ndarray, index: np.ndarray, embeddings: np.ndarray) -> None:
    """Write the three files of OpenLex3D's prediction format for one scene into path."""
    import open3d as o3d

    path.mkdir(parents=True, exist_ok=True)
    for old in [*path.glob("*.pcd"), *path.glob("*.ply")]:  # OpenLex3D reads the first cloud file it finds
        old.unlink()
    cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(points.astype(np.float32)))
    if not o3d.t.io.write_point_cloud(str(path / "point_cloud.pcd"), cloud):
        raise OSError(f"could not write {path / 'point_cloud.pcd'}")
    np.save(path / "index.npy", index.astype(np.int64))
    np.save(path / "embeddings.npy", embeddings.astype(np.float32))


def export(runs: list[Run], variant: str) -> tuple[str, dict]:
    """Export the runs' maps (one run per scene) as one method; returns its name and what to know about each scene."""
    method = method_name(runs[0], variant)
    about = {}
    for run in runs:
        scene = scene_name(replica.parse_source(run.source)[0])
        points, index, embeddings, about[scene] = prediction(run, variant)
        write_prediction(PREDICTIONS / method / "replica" / scene, points, index, embeddings)
        a = about[scene]
        print(f"{method} {scene}: {a['objects']} objects, {a['points']} points, ATE {100 * a['ate_m']:.1f} cm")
    return method, about


class SiglipText:
    """OpenLex3D's VisualLanguageEncoder interface over models/siglip.py's text encoder (registered by register)."""

    def __init__(self):
        from models import siglip

        self.encoder = siglip.load_encoder()

    def compute_text_features(self, input_text: list[str], batch_size: int = 64) -> np.ndarray:
        return np.concatenate(
            [
                self.encoder.encode_text(list(input_text[i : i + batch_size])).cpu().numpy()
                for i in range(0, len(input_text), batch_size)
            ]
        ).astype(np.float32)


def text_model(model_cfg):
    """The text encoder of OpenLex3D's model config, loaded once per process.

    OpenLex3D's evaluate_queries would load it again for every scene.
    """
    return load_text_model(model_cfg.type, model_cfg.backbone, model_cfg.get("checkpoint"), model_cfg.device)


@cache  # called with positional arguments only: a keyword call is another key and loads a second copy
def load_text_model(kind: str, backbone: str, checkpoint: str | None, device: str):
    from omegaconf import OmegaConf
    from openlex3d.models import load_model

    if kind == SIGLIP:
        return SiglipText()
    return load_model(
        OmegaConf.create({"type": kind, "backbone": backbone, "checkpoint": checkpoint, "device": device})
    )


def register() -> None:
    """Make OpenLex3D's code use text_model, with SigLIP 2 as model type SIGLIP, and faiss on the CPU.

    openlex3d.models.load_model imports openlex3d.models.<type>, which finds a module already in sys.modules. Its
    align_masks module takes faiss for the GPU build whenever faiss.contrib.torch_utils imports, which faiss-cpu has.
    """
    import faiss
    import openlex3d.core.align_masks as align_masks
    import openlex3d.core.cosine_similarity as cosine_similarity
    from openlex3d.models.base import VisualLanguageEncoder

    VisualLanguageEncoder.register(SiglipText)
    module = types.ModuleType(f"openlex3d.models.{SIGLIP}")
    module.load_model = lambda backbone=None, checkpoint=None, device="cuda": load_text_model(SIGLIP, "", None, "")
    sys.modules[module.__name__] = module
    cosine_similarity.load_model = text_model
    align_masks.use_gpu = hasattr(faiss, "StandardGpuResources")


def config(name: str, method: str, predictions: Path, output: Path, model: str, overrides: list[str]):
    """OpenLex3D's hydra config name (eval_segmentation or eval_query) for a method, with the paths of this project."""
    from hydra import compose, initialize_config_dir
    from openlex3d import get_path

    paths = {
        "replica_path": DATA_DIR / "vmap",
        "openlex_gt_path": DATA_DIR / "gt_openlex3d",
        "base_prediction_path": predictions,
        "output_path": output,
        "scannetpp_path": "",
        "hm3d_path": "",
    }
    model_overrides = (
        [f"model.type={SIGLIP}", "model.backbone=siglip2", "model.checkpoint=null"] if model == SIGLIP else []
    )
    with initialize_config_dir(config_dir=str(get_path() / "config"), version_base=None):
        return compose(
            config_name=name,
            overrides=[
                *(f"paths.{k}={v}" for k, v in paths.items()),
                f"evaluation.algorithm={method}",
                *model_overrides,
                *overrides,
            ],
        )


def score_segmentation(method: str, predictions: Path, model: str, scenes: list[str], topn: int, ranking: bool) -> dict:
    """OpenLex3D's segmentation scores of a method: per scene, and their mean over the scenes."""
    import openlex3d.core.metric as metric
    from openlex3d.core.evaluation import compute_feature_to_prompt_similarity, get_label_from_logits
    from openlex3d.core.io import load_predicted_features, load_prompt_list
    from openlex3d.datasets import load_dataset, load_openlex3d_cloud

    per_scene = {}
    for scene in scenes:
        output = RESULTS / "segmentation"
        overrides = ["dataset=segmentation/replica", f"dataset.scene={scene}", f"evaluation.topn={topn}"]
        cfg = config("eval_segmentation", method, predictions, output, model, overrides)
        gt_cloud, gt_ids, labels = load_dataset(cfg.dataset, load_openlex3d=True)
        visible = load_openlex3d_cloud(cfg.dataset.name, scene, cfg.dataset.openlex3d_path)
        if len(gt_cloud.point.positions) != len(visible.point.positions):  # load_dataset falls back to the mesh
            raise RuntimeError(f"{scene}: OpenLex3D did not load the visible GT cloud")
        pred_cloud, pred_feats = load_predicted_features(
            cfg.evaluation.predictions_path, cfg.evaluation.voxel_downsampling_size
        )
        prompts = load_prompt_list(cfg)
        logits = compute_feature_to_prompt_similarity(text_model(cfg.model), pred_feats, prompts)
        top = get_label_from_logits(logits, prompts, method="topn", topn=topn)
        excluded = cfg.evaluation.excluded_labels
        freq, *_ = metric.category_frequency_topn(pred_cloud, top, gt_cloud, gt_ids, labels, excluded)
        per_scene[scene] = {"freq": freq}
        if ranking:
            per_scene[scene]["ranking"] = metric.set_based_ranking(
                pred_cloud, gt_cloud, gt_ids, labels, excluded, logits, prompts
            )
        print(f"{method} {scene} top-{topn}: " + ", ".join(f"{k} {v:.3f}" for k, v in freq.items()))
    mean = {
        part: {
            k: float(np.mean([s[part][k] for s in per_scene.values()])) for k in next(iter(per_scene.values()))[part]
        }
        for part in next(iter(per_scene.values()))
    }
    return {"mean": mean, "per_scene": per_scene}


def score_queries(method: str, predictions: Path, model: str, scenes: list[str], level: str) -> dict:
    """OpenLex3D's query AP of a method at a query level: pooled over the scenes, and per scene."""
    from openlex3d.core.metric import compute_ap, compute_ap_averages
    from openlex3d.scripts.evaluate_queries import get_matches_for_scene

    # get_matches_for_scene saves the GT mesh and the matches of every scene, 60 MB each, for OpenLex3D's viewer
    with tempfile.TemporaryDirectory() as output:
        overrides = [
            "dataset=query/replica",
            f"dataset.scenes=[{','.join(scenes)}]",
            f"evaluation.query_level={level}",
            "evaluation.metric=ap",
            "evaluation.criteria=top_k",
            f"evaluation.top_k={TOP_K}",
        ]
        cfg = config("eval_query", method, predictions, Path(output), model, overrides)
        matches, per_scene = {}, {}
        for scene in scenes:
            scene_matches, _ = get_matches_for_scene(cfg, scene)
            matches.update(slim(scene_matches))
            per_scene[scene] = to_float(compute_ap_averages(compute_ap(scene_matches)[0]))
    overall = to_float(compute_ap_averages(compute_ap(matches)[0]))
    print(f"{method} queries {level}: " + ", ".join(f"{AP_KEYS[k]} {v:.2f}" for k, v in overall.items()))
    return {"overall": overall, "per_scene": per_scene}


def slim(matches: dict) -> dict:
    """OpenLex3D's query matches of a scene without what compute_ap doesn't read, to pool the scenes in less memory.

    Its matching deep-copies every overlapping GT object and prediction into the other's list, mesh vertex indices and
    lists of matches included. Held for 8 scenes, that took one process past 24 GB and the machine out of memory.
    """
    nested = {"mask_indices", "matched_pred", "matched_gt"}
    for scene in matches.values():
        for gt in scene["gt"]["object"]:
            gt.pop("mask_indices", None)
            gt["matched_pred"] = [{k: v for k, v in p.items() if k not in nested} for p in gt["matched_pred"]]
        for pred in scene["pred"]["object"]:
            pred.pop("mask_indices", None)
            pred["matched_gt"] = [{k: v for k, v in g.items() if k not in nested} for g in pred["matched_gt"]]
    return matches


def to_float(d: dict) -> dict:
    return {k: float(v) for k, v in d.items()}


def score(method: str, predictions: Path, model: str, scenes: list[str], args, about: dict | None = None) -> dict:
    """Run the tasks on a method's predictions under predictions/<method>/replica, saving the scores after each."""
    register()
    path = RESULTS / f"{method}.json"
    results = json.loads(path.read_text()) if path.exists() else {}  # keep the tasks this run doesn't redo
    results.update({"method": method, "model": model, "scenes": scenes})
    if about is not None:
        results["about"] = about

    def save():
        RESULTS.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))

    if "segmentation" in args.tasks:
        results.setdefault("segmentation", {})
        for n in args.topn:
            results["segmentation"][f"top{n}"] = score_segmentation(
                method, predictions, model, scenes, n, args.set_ranking
            )
            save()
    if "queries" in args.tasks:
        results.setdefault("queries", {})
        for level in args.levels:
            results["queries"][level] = score_queries(method, predictions, model, scenes, level)
            save()
    return results


def tables(topn: int) -> str:
    """Every method scored so far (RESULTS): the top-n segmentation frequencies, and the query AP of each level.

    The queries table starts with the paper's own rows (PAPER_QUERIES), which have all queries only.
    """
    results = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    paper = [
        {"method": f"{name} (paper)", "queries": {"all": {"overall": dict(zip(AP_KEYS, ap, strict=True))}}}
        for name, ap in PAPER_QUERIES.items()
    ]
    width = max([len(r["method"]) for r in results + paper]) + 2
    lines = [f"Segmentation, top-{topn} (share of the GT objects' points, mean over scenes)"]
    lines.append(f"{'':<{width}}" + "".join(f"{h:>9}" for h in CATEGORY_COLUMNS.values()) + "   scenes")
    for r in results:
        seg = r.get("segmentation", {}).get(f"top{topn}")
        if seg:
            freq = seg["mean"]["freq"]
            cells = "".join(f"{freq[k]:>9.3f}" for k in CATEGORY_COLUMNS)
            lines.append(f"{r['method']:<{width}}{cells}   {len(seg['per_scene'])}")
    lines += ["", f"Queries, AP in percent (top {TOP_K} per query, pooled over scenes)"]
    lines.append(f"{'':<{width}}" + "".join(f"{f'{level} {h}':>11}" for level in LEVELS for h in AP_KEYS.values()))
    for r in paper + results:
        queries = r.get("queries", {})
        if queries:
            cells = "".join(
                f"{queries[level]['overall'][k]:>11.2f}" if level in queries else f"{'-':>11}"
                for level in LEVELS
                for k in AP_KEYS
            )
            lines.append(f"{r['method']:<{width}}{cells}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="+", default=replica.SCENES, choices=replica.SCENES)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--mask-source", default="sam3", choices=["sam3", "gt"])
    parser.add_argument("--pose-source", default="gt", choices=["gt", "vggt"])
    parser.add_argument("--vocab", help="as for run.py: the SAM prompts the runs were made with")
    parser.add_argument("--variants", nargs="*", default=["graph"], choices=list(VARIANTS))
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=[],
        help=f"methods in --baseline-dir to score instead of our maps, such as concept-graphs hovsg (in {BASELINES})",
    )
    parser.add_argument("--baseline-dir", type=Path, default=BASELINES)
    parser.add_argument("--tasks", nargs="+", default=["segmentation", "queries"], choices=["segmentation", "queries"])
    parser.add_argument("--topn", nargs="+", type=int, default=[5], help="segmentation: the top-n labels per point")
    parser.add_argument("--set-ranking", action="store_true", help="segmentation: also the set ranking scores (slow)")
    parser.add_argument("--levels", nargs="+", default=list(LEVELS), choices=LEVELS)
    parser.add_argument("--skip-export", action="store_true", help="score the maps exported before")
    parser.add_argument("--table", action="store_true", help="only print the scores saved so far")
    args = parser.parse_args()
    scenes = [scene_name(s) for s in args.scenes]

    if not args.table:
        for method in args.baselines:
            score(method, args.baseline_dir, "open_clip", scenes, args)
        if not args.baselines:
            runs = [
                footage.make_run(
                    f"replica/{s}/00", args.stride, args.mask_source, args.pose_source, vocab.tag(args.vocab)
                )
                for s in args.scenes
            ]
            for variant in args.variants:
                method, about = method_name(runs[0], variant), None
                if not args.skip_export:
                    method, about = export(runs, variant)
                score(method, PREDICTIONS, SIGLIP, scenes, args, about)
    print()
    print(tables(args.topn[0]))


if __name__ == "__main__":
    main()
