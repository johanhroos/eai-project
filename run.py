"""Run the pipeline on Replica sequences or phone videos. Each stage is cached, so reruns compute only what is missing.

Replica runs are scored against the ground truth at the end. Phone runs (--source phone/<name> for a video
data/phone/<name>.<ext>) have no ground truth: they use VGGT-Omega's poses and stop after the graph.

The graph stage (graph/build.py) takes the masks one frame at a time. A mask joins the object its SAM track
belongs to; new tracks, and tracks that jumped to another object, join the object in memory they match by DINO
appearance and location (graph/memory.py), or make a new one. At the end, nodes whose points form clusters
apart are split, each is labelled from all its views, masks and parts of one object are merged, and only the objects
whose box can be trusted are kept (graph/refine.py); each object's box is then drawn from the points its views agree
on (graph/multiview.py).
"""

import argparse
import json
import os
from dataclasses import replace
from functools import partial

# Lets PyTorch reuse reserved but free GPU memory for large blocks: SAM 3's tracker ran out of a 10 GB GPU on phone
# video with 2.7 GB reserved and unused. Set before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from data import cli, footage, replica, vocab
from data.stages import (
    Boxes,
    Events,
    Features,
    Graph,
    Masks,
    ObjectBoxes,
    Run,
    TextAligned,
    check_aligned,
    load_stage,
    mask_vocab,
    save_stage,
)
from evaluation import ground_truth, metrics
from geometry import mask_boxes
from graph import build
from models import dino, sam, siglip, vggt

STAGES = ["frames", "masks", "features", "siglip", "vggt", "boxes", "graph", "events", "gt_boxes"]
# A stage is recomputed when one of these was saved after it, such as boxes after a new vggt stage
UPSTREAM = {"masks": ["frames"], "features": ["masks"], "siglip": ["masks"], "boxes": ["masks", "vggt"]}
UPSTREAM["graph"] = ["masks", "features", "siglip", "boxes"]

parser = argparse.ArgumentParser(description=__doc__)
cli.add_run_arguments(parser, source=cli.REPLICA)
parser.add_argument(
    "--graph-only",
    action="store_true",
    help="only rebuild the graph from the cached stages, without loading any model or scoring it",
)
parser.add_argument("--skip-text-eval", action="store_true", help="score identities without loading the text encoder")
parser.add_argument(
    "--overwrite",
    nargs="*",
    choices=STAGES,
    help="recompute these stages even if they are cached, or every stage when none is named",
)
args = parser.parse_args()
runs = cli.runs(parser, args)


def stale(run: Run, stage: str, params: dict | None = None) -> bool:
    """Whether to compute the stage.

    It is missing, --overwrite names it (or names no stage), its metadata.json holds other values for params, or a
    stage it is computed from (UPSTREAM) was saved after it.
    """
    if args.overwrite is not None and (not args.overwrite or stage in args.overwrite):
        return True
    if not run.done(stage):
        return True
    if stage == "graph" and not all((run.stage_dir(stage) / f).exists() for f in ("matches.json", "history.json")):
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
    for load in (sam.load_model, dino.load_encoder, siglip.load_encoder, vggt.load_model, vggt.load_depth_model):
        load.cache_clear()
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
        build.rebuild(run)
        node = load_stage(run.stage_dir("graph"), Graph).node
        print(f"{source}: saved {len(np.unique(node[node >= 0]))} objects to {run.stage_dir('graph')}")
        continue
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
        save_stage(run.stage_dir("features"), dino.features(run, masks, dino.load_encoder()), {"model": dino.MODEL_ID})
        check_aligned(run, "features")
        print(f"Saved features to {run.stage_dir('features')}")
    features = load_stage(run.stage_dir("features"), Features)

    if stale(run, "siglip"):
        embeddings = siglip.features(run, masks, siglip.load_encoder())
        save_stage(run.stage_dir("siglip"), embeddings, {"model": siglip.MODEL_ID})
        check_aligned(run, "siglip")
        print(f"Saved SigLIP embeddings to {run.stage_dir('siglip')}")

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

    if stale(run, "boxes", {"points_per_mask": mask_boxes.POINTS_PER_MASK, "conf": "median"}):
        meta = {"pose_source": run.pose_source, "points_per_mask": mask_boxes.POINTS_PER_MASK, "conf": "median"}
        save_stage(run.stage_dir("boxes"), mask_boxes.compute_boxes(run, masks), meta)
        check_aligned(run, "boxes")
        print(f"Saved boxes to {run.stage_dir('boxes')}")
    boxes = load_stage(run.stage_dir("boxes"), Boxes)

    if stale(run, "graph", build.metadata()):
        names = mask_vocab(run)
        embeddings = load_stage(run.stage_dir("siglip"), TextAligned)
        built = build.replay(masks, boxes, features, embeddings, names, frames=footage.frame_numbers(run))
        finish = partial(build.finish, run, masks, boxes, features, embeddings, names)
        build.save_result(run.stage_dir("graph"), built, finish)
        check_aligned(run, "graph")
        node = load_stage(run.stage_dir("graph"), Graph).node
        print(f"Saved graph to {run.stage_dir('graph')}: {len(np.unique(node[node >= 0]))} objects")

    if not footage.is_replica(run):
        node = load_stage(run.stage_dir("graph"), Graph).node
        clip = footage.clip_index(run)[np.searchsorted(footage.frame_numbers(run), masks.frame)]
        tracks, objects = len(np.unique(masks.sam_id)), len(np.unique(node[node >= 0]))
        print(f"{run.source}: {len(masks.frame)} masks, {tracks} SAM tracks, {objects} objects")
        if clip.max() > 0:
            shared = sum(len(np.unique(clip[node == n])) > 1 for n in np.unique(node[node >= 0]))
            print(f"  {shared} of the objects are seen in more than one video")
        continue  # no ground truth to score against

    # Ground truth for the evaluation: always from GT masks and poses, whatever mask_source and pose_source are
    if stale(run, "events"):
        meta = {"min_pixels": ground_truth.MIN_PIXELS, "min_run": ground_truth.MIN_RUN, "min_gap": ground_truth.MIN_GAP}
        save_stage(run.stage_dir("events"), ground_truth.compute_events(run), {**meta, "kinds": ground_truth.KIND})
        print(f"Saved events to {run.stage_dir('events')}")

    # FOR EVALUATION: always from GT masks and poses, whatever mask_source and pose_source are
    if stale(run, "gt_boxes"):
        save_stage(run.stage_dir("gt_boxes"), ground_truth.compute_gt_boxes(run), {})
        print(f"Saved GT boxes to {run.stage_dir('gt_boxes')}")

    # Evaluation: cheap, so always recomputed. The raw SAM tracks show what the graph adds.
    # With --mask-source gt the tracks are the GT objects: a perfect map, so their query scores are what SigLIP can do.
    gt_run = replace(run, mask_source="gt", vocab="")
    if not gt_run.done("masks"):
        compute_masks(gt_run)
    gt = load_stage(gt_run.stage_dir("masks"), Masks)
    events = load_stage(run.stage_dir("events"), Events)
    objects = replica.load_objects(replica.parse_source(run.source)[0])
    graphs = {"tracks, no graph": Graph(masks.sam_id), "graph": load_stage(run.stage_dir("graph"), Graph)}
    predictions = {name: graph.node for name, graph in graphs.items()}
    results[source] = {name: metrics.evaluate(masks, ids, gt, events, objects) for name, ids in predictions.items()}
    embeddings = load_stage(run.stage_dir("siglip"), TextAligned)
    for level in () if args.skip_text_eval else ("level0", "level1"):
        targets = replica.load_queries(replica.parse_source(run.source)[0], level)
        text = siglip.load_encoder().encode_text(list(targets)).cpu().numpy()
        for name, ids in predictions.items():
            scores = metrics.queries(masks, ids, gt, embeddings, targets, text)
            results[source][name].update({f"{level}_{k}": v for k, v in scores.items()})
    names = mask_vocab(run)
    gt_boxes = load_stage(run.stage_dir("gt_boxes"), ObjectBoxes)
    for name, graph in graphs.items():
        results[source][name].update(metrics.nearest_objects(masks, graph, gt, boxes, names, gt_boxes, objects))
    run.stage_dir("eval").mkdir(parents=True, exist_ok=True)
    (run.stage_dir("eval") / "metrics.json").write_text(json.dumps(results[source], indent=2))

if results:
    print(f"\nmask source {args.mask_source}, pose source {runs[0].pose_source}, stride {args.stride}")
    print(metrics.table(results))
    if not args.skip_text_eval:
        print("\nText queries (OpenLex3D level 0 and level 1)")
        print(metrics.table(results, metrics.QUERY_COLUMNS))
    print("\nNearest object of a type (graph.query --near), against the GT boxes")
    print(metrics.table(results, metrics.NEAR_COLUMNS))
