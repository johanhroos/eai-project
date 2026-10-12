"""Run the pipeline on Replica sequences or phone videos. Each stage is cached, so reruns compute only what is missing.

Replica runs are scored against the ground truth at the end. Phone runs (--source phone/<name> for a video
data/phone/<name>.<ext>) have no ground truth: they use VGGT-Omega's poses and stop after the graph.
"""

import argparse
import json
from dataclasses import replace

import numpy as np
import torch

import evaluate
from graph import pipeline, query
from graph.association import AssociationConfig
from models import dino, sam, siglip
from utils import footage, lift3d, replica, vocab
from utils.schema import (
    Boxes,
    Events,
    Features,
    Graph,
    Masks,
    Run,
    TextAligned,
    check_aligned,
    load_stage,
    save_stage,
)

STAGES = ["frames", "masks", "features", "siglip", "vggt", "boxes", "graph", "events", "gt_boxes"]
# A stage is recomputed when one of these was saved after it, such as boxes after a new vggt stage
UPSTREAM = {"masks": ["frames"], "features": ["masks"], "siglip": ["masks"], "boxes": ["masks", "vggt"]}
UPSTREAM["graph"] = ["masks", "features", "siglip", "boxes"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--source",
    nargs="+",
    default=[f"replica/{scene}/00" for scene in replica.SCENES],
    help="runs to compute, such as replica/room_0/00 or phone/kitchen (default: every Replica scene's sequence 00)",
)
parser.add_argument("--stride", type=int, default=20)
parser.add_argument("--mask-source", default="sam3", choices=["sam3", "gt"])
parser.add_argument("--association-config", type=str, default=str(pipeline.DEFAULT_CONFIG))
parser.add_argument(
    "--graph-only", action="store_true", help="rebuild only from cached observations; no model inference"
)
parser.add_argument("--skip-text-eval", action="store_true", help="score identities without loading the text encoder")
parser.add_argument(
    "--pose-source", choices=["gt", "vggt"], help="default: Replica's on Replica, VGGT-Omega's on phone video"
)
parser.add_argument(
    "--vocab",
    help="SAM prompts: a name in vocab.NAMED such as indoor, or a text file with one prompt per line. Default: the "
    "scene's GT classes on Replica, vocab.INDOOR on phone video. Its outputs go to their own directories",
)
parser.add_argument(
    "--overwrite",
    nargs="*",
    choices=STAGES,
    help="recompute these stages even if they are cached, or every stage when none is named",
)
args = parser.parse_args()
try:
    association_config = AssociationConfig.from_json(args.association_config)
except (ValueError, TypeError, OSError) as e:
    parser.error(f"invalid association configuration: {e}")
if args.stride < 1:
    parser.error("stride must be positive")
try:
    runs = [
        footage.make_run(s, args.stride, args.mask_source, args.pose_source, vocab.tag(args.vocab)) for s in args.source
    ]
except ValueError as e:
    parser.error(str(e))


def stale(run: Run, stage: str, params: dict | None = None) -> bool:
    """Whether to compute the stage.

    It is missing, --overwrite names it (or names no stage), its metadata.json holds other values for params, or a
    stage it is computed from (UPSTREAM) was saved after it.
    """
    if args.overwrite is not None and (not args.overwrite or stage in args.overwrite):
        return True
    if not run.done(stage):
        return True
    if stage == "graph" and (
        not (run.stage_dir("graph") / "memory" / "metadata.json").exists()
        or any(not (run.stage_dir("graph") / f"{name}.json").exists() for name in ("edges", "matches", "history"))
    ):
        return True
    metadata = run.stage_dir(stage) / "metadata.json"
    cached = json.loads(metadata.read_text())
    if params is not None and {k: cached.get(k) for k in params} != params:
        print(f"Recomputing {run.stage_dir(stage)}: cached with other parameters")
        return True
    for up in UPSTREAM.get(stage, []):
        used = (up != "frames" or not footage.is_replica(run)) and (up != "vggt" or run.pose_source == "vggt")
        if used and (run.stage_dir(up) / "metadata.json").stat().st_mtime > metadata.stat().st_mtime:
            print(f"Recomputing {run.stage_dir(stage)}: {up} is newer")
            return True
    print(f"Skipping {run.source}: {run.stage_dir(stage)} exists")
    return False


def prompts(run: Run) -> list[str]:
    """SAM's text prompts for the run: --vocab, else the scene's GT classes on Replica and vocab.INDOOR otherwise."""
    if args.vocab:
        return vocab.prompts(args.vocab)
    return replica.prompt_vocab(run) if footage.is_replica(run) else vocab.INDOOR


def free_gpu() -> None:
    """Drop every loaded model, so VGGT-Omega gets the GPU to itself and the next run's SAM gets it back."""
    for load in (sam.load_model, dino.load_encoder, siglip.load_encoder):
        load.cache_clear()
    if "vggt" in globals():
        vggt.load_model.cache_clear()
        vggt.load_depth_model.cache_clear()
    torch.cuda.empty_cache()


def compute_masks(run: Run) -> None:
    if run.mask_source == "gt":
        masks, names = replica.gt_masks(run)
        metadata = {"vocab": names}
    else:
        frames, numbers = footage.load_run_rgb(run)
        names = prompts(run)
        processor, model = sam.load_model()
        masks = sam.segment_clips(frames, numbers, footage.clip_index(run), names, processor, model)
        metadata = {
            "vocab": names,
            "model": sam.MODEL_ID,
            "score_threshold_detection": model.config.score_threshold_detection,
            "new_det_thresh": model.config.new_det_thresh,
        }
    save_stage(run.stage_dir("masks"), masks, metadata)
    print(f"Saved {len(masks.frame)} masks to {run.stage_dir('masks')}")


results = {}
for run in runs:
    source = run.source
    if args.graph_only:
        result = pipeline.rebuild(run, association_config)
        print(f"{source}: saved {len(result.memory.nodes)} persistent objects to {run.stage_dir('graph')}")
        continue
    if run.pose_source == "vggt":
        from models import vggt
    if not footage.is_replica(run) and stale(run, "frames"):
        footage.extract(run)
        print(f"Saved {len(footage.frame_numbers(run))} frames to {run.stage_dir('frames')}")
    n_frames = len(footage.frame_numbers(run))
    if run.pose_source == "vggt" and n_frames > vggt.MAX_FRAMES:  # check before hours of SAM, not after
        stride = -(-run.stride * n_frames // vggt.MAX_FRAMES)
        raise SystemExit(f"{source}: {n_frames} frames, but VGGT-Omega takes {vggt.MAX_FRAMES}: use --stride {stride}")

    if stale(run, "masks", {"vocab": prompts(run)} if run.mask_source == "sam3" else None):
        compute_masks(run)
    masks = load_stage(run.stage_dir("masks"), Masks)

    if stale(run, "features"):
        free_gpu()
        save_stage(run.stage_dir("features"), dino.features(run, masks, dino.load_encoder()), {"model": dino.MODEL_ID})
        check_aligned(run, "features")
        print(f"Saved features to {run.stage_dir('features')}")
    features = load_stage(run.stage_dir("features"), Features)

    if stale(run, "siglip"):
        free_gpu()
        embeddings = siglip.features(run, masks, siglip.load_encoder())
        save_stage(run.stage_dir("siglip"), embeddings, {"model": siglip.MODEL_ID})
        check_aligned(run, "siglip")
        print(f"Saved SigLIP embeddings to {run.stage_dir('siglip')}")
    embeddings = load_stage(run.stage_dir("siglip"), TextAligned)

    # Depth and cameras from VGGT-Omega instead of Replica's, in metres with z up and without ground truth
    if run.pose_source == "vggt" and stale(run, "vggt", vggt.PARAMS):
        free_gpu()
        geometry = vggt.reconstruct(run, vggt.load_model())
        errors = vggt.gt_errors(run, geometry) if footage.is_replica(run) else {}
        save_stage(run.stage_dir("vggt"), geometry, {**vggt.PARAMS, "scale": geometry.scale, **errors})
        print(f"Saved VGGT-Omega depth and cameras to {run.stage_dir('vggt')}: {geometry.scale:.2f} m per unit")
        if errors:
            print(
                f"  against the GT trajectory: ATE {100 * errors['ate_m']:.1f} cm, scale off by "
                f"{100 * errors['scale_error']:+.1f}%, up off by {errors['up_error_deg']:.2f} degrees"
            )
        free_gpu()

    if stale(run, "boxes", {"points_per_mask": lift3d.POINTS_PER_MASK}):
        meta = {"pose_source": run.pose_source, "points_per_mask": lift3d.POINTS_PER_MASK}
        save_stage(run.stage_dir("boxes"), lift3d.compute_boxes(run, masks), meta)
        check_aligned(run, "boxes")
        print(f"Saved boxes to {run.stage_dir('boxes')}")
    boxes = load_stage(run.stage_dir("boxes"), Boxes)

    if stale(run, "graph", association_config.metadata()):
        names = json.loads((run.stage_dir("masks") / "metadata.json").read_text())["vocab"]
        built = pipeline.replay(
            masks, boxes, features, embeddings, names, association_config, frames=footage.frame_numbers(run)
        )
        pipeline.save_result(run.stage_dir("graph"), built)
        check_aligned(run, "graph")
        print(f"Saved graph to {run.stage_dir('graph')}")

    if not footage.is_replica(run):
        node = load_stage(run.stage_dir("graph"), Graph).node
        clip = footage.clip_index(run)[np.searchsorted(footage.frame_numbers(run), masks.frame)]
        tracks, objects = len(np.unique(masks.sam_id)), len(np.unique(node))
        print(f"{run.source}: {len(masks.frame)} masks, {tracks} SAM tracks, {objects} objects")
        if clip.max() > 0:
            shared = sum(len(np.unique(clip[node == n])) > 1 for n in np.unique(node))
            print(f"  {shared} of the objects are seen in more than one video")
        continue  # no ground truth to score against

    # Ground truth for the evaluation: always from GT masks and poses, whatever mask_source and pose_source are
    if stale(run, "events"):
        meta = {"min_pixels": lift3d.MIN_PIXELS, "min_run": lift3d.MIN_RUN, "min_gap": lift3d.MIN_GAP}
        save_stage(run.stage_dir("events"), lift3d.compute_events(run), {**meta, "kinds": lift3d.KIND})
        print(f"Saved events to {run.stage_dir('events')}")

    # FOR EVALUATION: always from GT masks and poses, whatever mask_source and pose_source are
    if stale(run, "gt_boxes"):
        save_stage(run.stage_dir("gt_boxes"), lift3d.compute_gt_boxes(run), {})
        print(f"Saved GT boxes to {run.stage_dir('gt_boxes')}")

    # Evaluation: cheap, so always recomputed. The raw SAM tracks show what the graph adds.
    # With --mask-source gt the tracks are the GT objects: a perfect map, so their query scores are what SigLIP can do.
    gt_run = replace(run, mask_source="gt", vocab="")
    if not gt_run.done("masks"):
        compute_masks(gt_run)
    gt = load_stage(gt_run.stage_dir("masks"), Masks)
    events = load_stage(run.stage_dir("events"), Events)
    objects = replica.load_objects(replica.parse_source(run.source)[0])
    predictions = {
        "tracks, no graph": masks.sam_id,
        "graph": load_stage(run.stage_dir("graph"), Graph).node,
    }
    results[source] = {name: evaluate.evaluate(masks, ids, gt, events, objects) for name, ids in predictions.items()}
    embeddings = load_stage(run.stage_dir("siglip"), TextAligned)
    persistent_vectors = query.memory_embeddings(pipeline.load_memory(run.stage_dir("graph") / "memory"))
    for level in () if args.skip_text_eval else ("level0", "level1"):
        targets = replica.load_queries(replica.parse_source(run.source)[0], level)
        text = siglip.load_encoder().encode_text(list(targets)).cpu().numpy()
        for name, ids in predictions.items():
            scores = evaluate.queries(
                masks,
                ids,
                gt,
                embeddings,
                targets,
                text,
                object_vectors=persistent_vectors if name == "graph" else None,
            )
            results[source][name].update({f"{level}_{k}": v for k, v in scores.items()})
    run.stage_dir("eval").mkdir(parents=True, exist_ok=True)
    (run.stage_dir("eval") / "metrics.json").write_text(json.dumps(results[source], indent=2))

if results:
    print(f"\nmask source {args.mask_source}, pose source {runs[0].pose_source}, stride {args.stride}")
    print(evaluate.table(results))
    if not args.skip_text_eval:
        print("\nText queries (OpenLex3D level 0 and level 1)")
        print(evaluate.table(results, evaluate.QUERY_COLUMNS))
