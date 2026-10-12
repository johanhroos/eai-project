"""Text queries over the object map: every object gets one SigLIP embedding, and a query ranks them by cosine.

Run it to search a run's graph, after run.py has cached the stages:

    uv run --no-sync python -m graph.query --source replica/room_0/00 "striped pillow"
    uv run --no-sync python -m graph.query --source phone/kitchen "coffee mug"
"""

import numpy as np

from utils.schema import Masks, TextAligned


def object_embeddings(masks: Masks, emb: TextAligned, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct object IDs in ids and their (M, D) L2-normalized embeddings, the mean over each one's masks.

    ids gives one object ID per mask, such as Graph.node or Masks.sam_id. For the graph on SAM 3 masks, a plain mean
    ranked the OpenLex3D queries better than one weighted by mask area (L0 mAP 0.44 against 0.41 over the 8 scenes).
    """
    objects, inverse = np.unique(ids, return_inverse=True)
    sums = np.zeros((len(objects), emb.siglip.shape[1]), np.float32)
    np.add.at(sums, inverse, emb.siglip.astype(np.float32))
    return objects, sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-6)


def rank(text_emb: np.ndarray, object_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each query, the object rows from best to worst match and their cosine similarities, both (Q, M).

    text_emb is (Q, D) from SiglipEncoder.encode_text and object_emb (M, D) from object_embeddings.
    """
    scores = text_emb @ object_emb.T
    order = np.argsort(-scores, axis=1, kind="stable")
    return order, np.take_along_axis(scores, order, axis=1)


if __name__ == "__main__":
    import argparse
    import json

    from graph.node import build_nodes
    from models import siglip
    from utils import footage, vocab
    from utils.schema import Boxes, Features, Graph, load_stage

    parser = argparse.ArgumentParser(description="Print the graph nodes that best match a text.")
    parser.add_argument("text", nargs="+", help="what to look for, such as: striped pillow")
    parser.add_argument("--source", default="replica/room_0/00")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--mask-source", default="sam3", choices=["sam3", "gt"])
    parser.add_argument("--pose-source", choices=["gt", "vggt"], help="as for run.py")
    parser.add_argument("--vocab", help="as for run.py")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    try:
        run = footage.make_run(args.source, args.stride, args.mask_source, args.pose_source, vocab.tag(args.vocab))
    except ValueError as e:
        parser.error(str(e))
    masks = load_stage(run.stage_dir("masks"), Masks)
    vocab = json.loads((run.stage_dir("masks") / "metadata.json").read_text())["vocab"]
    graph = load_stage(run.stage_dir("graph"), Graph)
    nodes = build_nodes(
        masks,
        load_stage(run.stage_dir("boxes"), Boxes),
        load_stage(run.stage_dir("features"), Features),
        graph,
        vocab,
    )
    objects, object_emb = object_embeddings(masks, load_stage(run.stage_dir("siglip"), TextAligned), graph.node)
    text = " ".join(args.text)
    order, scores = rank(siglip.load_encoder().encode_text([text]).cpu().numpy(), object_emb)

    print(f'{run.source}: "{text}"')
    for row, score in zip(order[0, : args.top], scores[0, : args.top], strict=True):
        n = nodes[int(objects[row])]  # build_nodes lists nodes in ID order, and IDs run 0, 1, ...
        centre = ", ".join(f"{c:.2f}" for c in (n.box_min + n.box_max) / 2)
        print(
            f"  {score:.3f}  node {n.id:>3} {n.label:<14} centre ({centre}) m, frames {n.frames[0]}-{n.frames[-1]}"
            f" ({len(n.frames)} seen), SAM tracks {n.sam_ids}"
        )
