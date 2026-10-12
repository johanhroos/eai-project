"""Text queries over the object map: every object gets one SigLIP embedding, and a query ranks them by cosine.

Run it to search a run's graph, after run.py has cached the stages:

    uv run --no-sync python -m graph.query --source replica/room_0/00 "striped pillow"
    uv run --no-sync python -m graph.query --source phone/kitchen "coffee mug"

--near ranks the matches by distance instead, to the object that best matches another text (graph/edges.py):

    uv run --no-sync python -m graph.query --source replica/room_0/00 cushion --near "grey sofa"
"""

import numpy as np

from utils.schema import Masks, TextAligned


def memory_embeddings(memory):
    """Text-retrieval descriptors directly from the compact saved memory."""
    nodes = [n for n in memory.nodes.values() if n.siglip is not None]
    ids = np.array([n.id for n in nodes], np.int32)
    dimension = memory.siglip_dimension or 0
    embeddings = np.stack([n.siglip for n in nodes]) if nodes else np.empty((0, dimension), np.float32)
    return ids, embeddings


def nearest(memory, position, top=5):
    """Nearest valid object boxes to a world position, as (ID, distance) pairs."""
    from graph.association import valid_box

    point = np.asarray(position, np.float32)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("position must be a finite world-coordinate 3-vector")
    distances = []
    for node in memory.nodes.values():
        if valid_box(node.box):
            gap = np.maximum(np.maximum(node.box.min - point, point - node.box.max), 0)
            distances.append((node.id, float(np.linalg.norm(gap))))
    return sorted(distances, key=lambda item: (item[1], item[0]))[:top]


def object_embeddings(masks: Masks, emb: TextAligned, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct object IDs in ids and their (M, D) L2-normalized embeddings, the mean over each one's masks.

    ids gives one object ID per mask, such as Graph.node or Masks.sam_id. For the graph on SAM 3 masks, a plain mean
    ranked the OpenLex3D queries better than one weighted by mask area (L0 mAP 0.44 against 0.41 over the 8 scenes).
    """
    objects, inverse = np.unique(ids, return_inverse=True)
    sums = np.zeros((len(objects), emb.siglip.shape[1]), np.float32)
    np.add.at(sums, inverse, emb.siglip.astype(np.float32))
    return objects, sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-6)


def of_type(labels: list[str], text: str, scores: np.ndarray, top: int) -> np.ndarray:
    """Rows of the objects that are a text, for --near: the objects labelled with it when it is one of the labels.

    Otherwise the top best matches by SigLIP. Labels come first because SigLIP's scores for a type are close: on room_0,
    "lamp" scores 0.116 for the lamp and 0.112 for blinds. labels and scores (M,) are per object row.
    """
    rows = np.flatnonzero(np.array(labels) == text.strip().lower())
    return rows if len(rows) else np.argsort(-scores, kind="stable")[:top]


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

    from graph import edges as graph_edges
    from graph.edges import build_edges, neighbours
    from graph.node import build_nodes
    from graph.pipeline import load_memory
    from models import siglip
    from utils import footage, vocab
    from utils.schema import Boxes, Features, Graph, load_stage

    parser = argparse.ArgumentParser(description="Print the graph nodes that best match a text.")
    parser.add_argument("text", nargs="*", help="what to look for, such as: striped pillow")
    parser.add_argument(
        "--position", nargs=3, type=float, help="nearest objects to a world-coordinate x y z (implies --association)"
    )
    parser.add_argument(
        "--association",
        action="store_true",
        help="search the association stage's memory (graph/pipeline.py, run.py --association) instead of the graph",
    )
    parser.add_argument("--source", default="replica/room_0/00")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--mask-source", default="sam3", choices=["sam3", "gt"])
    parser.add_argument("--pose-source", choices=["gt", "vggt"], help="as for run.py")
    parser.add_argument("--vocab", help="as for run.py")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument(
        "--near",
        help="rank the matches by distance to the object that best matches this text, such as: sofa. The matches are "
        "then the objects labelled with the text when it is one of SAM's prompts, else its --top best matches",
    )
    args = parser.parse_args()
    if not args.text and args.position is None:
        parser.error("provide a text query or --position x y z")

    try:
        run = footage.make_run(args.source, args.stride, args.mask_source, args.pose_source, vocab.tag(args.vocab))
    except ValueError as e:
        parser.error(str(e))
    memory_path = run.stage_dir("association") / "memory"
    if (args.association or args.position is not None) and (memory_path / "metadata.json").exists():
        memory = load_memory(memory_path)
        if args.position is not None:
            for object_id, distance in nearest(memory, args.position, args.top):
                print(f"node {object_id}: {memory.nodes[object_id].label}, distance {distance:.2f} m")
        if args.text:
            objects, object_emb = memory_embeddings(memory)
            if len(objects):
                order, scores = rank(siglip.load_encoder().encode_text([" ".join(args.text)]).cpu().numpy(), object_emb)
                for row, score in zip(order[0, : args.top], scores[0, : args.top], strict=True):
                    node = memory.nodes[int(objects[row])]
                    print(f"{score:.3f} node {node.id}: {node.label}, seen {node.first_seen}-{node.last_seen}")
        raise SystemExit(0)
    if args.position is not None:
        parser.error("--position needs the association stage; build it with run.py --association")
    if args.association:
        parser.error("no association stage; build it with run.py --association")
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
    edges = build_edges(nodes)
    queries = [text] if args.near is None else [text, args.near]
    sims = siglip.load_encoder().encode_text(queries).cpu().numpy() @ object_emb.T  # (Q, M)
    # build_nodes lists nodes in ID order, and IDs run 0, 1, ..., so objects[row] indexes nodes

    def describe(n) -> str:
        centre = ", ".join(f"{c:.2f}" for c in (n.box_min + n.box_max) / 2)
        links = ", ".join(f"{rel} {nodes[o].label} {o}" for rel, o in neighbours(edges, n.id) if rel != "near")
        return (
            f"node {n.id:>3} {n.label:<14} centre ({centre}) m, frames {n.frames[0]}-{n.frames[-1]}"
            f" ({len(n.frames)} seen), SAM tracks {n.sam_ids}" + (f", {links}" if links else "")
        )

    if args.near is None:
        print(f'{run.source}: "{text}"')
        for row in np.argsort(-sims[0], kind="stable")[: args.top]:
            print(f"  {sims[0, row]:.3f}  {describe(nodes[int(objects[row])])}")
    else:
        anchor = nodes[int(objects[sims[1].argmax()])]
        rows = of_type([nodes[int(o)].label for o in objects], text, sims[0], args.top)
        print(f'{run.source}: "{text}" nearest to "{args.near}", {sims[1].max():.3f}  {describe(anchor)}')
        for node_id, gap, centre in graph_edges.nearest(nodes, anchor.id, [int(objects[r]) for r in rows])[: args.top]:
            print(f"  {gap:.2f} m apart, {centre:.2f} m between centres  {describe(nodes[node_id])}")
