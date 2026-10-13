"""The command-line arguments that pick a run, shared by run.py and every tool that reads its cached stages."""

import argparse

from data import footage, replica, vocab
from data.stages import Run

REPLICA = [f"replica/{scene}/00" for scene in replica.SCENES]  # every Replica scene's sequence 00


def add_run_arguments(parser: argparse.ArgumentParser, source: str | list[str] | None = "replica/room_0/00") -> None:
    """--source, --stride, --mask-source, --pose-source and --vocab.

    source is --source's default: a list lets it name several runs, and None makes it required.
    """
    if isinstance(source, list):
        parser.add_argument("--source", nargs="+", default=source, help="such as replica/room_0/00 or phone/kitchen")
    else:
        parser.add_argument("--source", default=source, required=source is None, help="such as phone/kitchen")
    parser.add_argument("--stride", type=int, default=20, help="use every stride-th frame")
    parser.add_argument("--mask-source", default="sam3", choices=["sam3", "gt"])
    parser.add_argument(
        "--pose-source", choices=["gt", "vggt"], help="default: Replica's on Replica, VGGT-Omega's on phone video"
    )
    parser.add_argument(
        "--vocab",
        help="SAM prompts: a name in vocab.NAMED such as indoor, or a text file with one prompt per line. Default: the "
        "scene's GT classes on Replica, vocab.INDOOR on phone video. Its outputs go to their own directories",
    )


def runs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[Run]:
    """The runs that args name; a phone video with GT masks or poses, which it doesn't have, is the parser's error."""
    if args.stride < 1:
        parser.error("stride must be positive")
    sources = args.source if isinstance(args.source, list) else [args.source]
    tag = vocab.tag(args.vocab)
    try:
        return [footage.make_run(s, args.stride, args.mask_source, args.pose_source, tag) for s in sources]
    except ValueError as e:
        parser.error(str(e))
