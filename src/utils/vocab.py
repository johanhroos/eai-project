"""Text prompts for SAM 3 on footage without ground truth: common indoor objects, one fixed list for every run.

Replica runs prompt with the scene's own GT classes by default (replica.prompt_vocab), which leaks the ground truth.
run.py --vocab replaces either default with a list named in NAMED or a text file of one prompt per line, and the run's
outputs then go to their own directories (Run.vocab). Walls, floors and ceilings are left out of INDOOR: they are
background, and their masks swallow the objects in front of them.
"""

from pathlib import Path

INDOOR = [
    "armchair",
    "backpack",
    "bag",
    "basket",
    "bed",
    "bench",
    "blanket",
    "blinds",
    "book",
    "bookshelf",
    "bottle",
    "bowl",
    "box",
    "cabinet",
    "ceiling light",
    "chair",
    "clock",
    "computer mouse",
    "counter",
    "cup",
    "curtain",
    "cushion",
    "desk",
    "door",
    "drawer",
    "fan",
    "kettle",
    "keyboard",
    "lamp",
    "laptop",
    "microwave",
    "mirror",
    "monitor",
    "mug",
    "oven",
    "phone",
    "picture",
    "pillow",
    "plant",
    "plate",
    "radiator",
    "refrigerator",
    "remote control",
    "rug",
    "shelf",
    "shoe",
    "sink",
    "sofa",
    "speaker",
    "stool",
    "stove",
    "table",
    "television",
    "toilet",
    "towel",
    "toy",
    "trash can",
    "vase",
    "wardrobe",
    "window",
]

NAMED = {"indoor": INDOOR}


def tag(arg: str | None) -> str:
    """The Run.vocab of a --vocab argument: the list's name or the file's stem, "" without one."""
    if not arg:
        return ""
    return arg if arg in NAMED else Path(arg).stem


def prompts(arg: str) -> list[str]:
    """The prompts of a --vocab argument: a name in NAMED or a text file with one prompt per line."""
    if arg in NAMED:
        return NAMED[arg]
    return [line.strip() for line in Path(arg).read_text().splitlines() if line.strip()]
