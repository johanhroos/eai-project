"""Text prompts for SAM 3 on footage without ground truth: common indoor objects, one fixed list for every run.

Replica runs prompt with the scene's own GT classes by default (replica.prompt_vocab), which leaks the ground truth.
run.py --vocab replaces either default with a list named in NAMED or a text file of one prompt per line, and the run's
outputs then go to their own directories (Run.vocab). Walls, floors and ceilings are left out of INDOOR: they are
background, and their masks swallow the objects in front of them.

INDOOR's things are found by name only: SAM 3 segments what a prompt names, and text search uses SigLIP, so a prompt
that is a synonym of another (coffee table, table) only costs SAM time. The list grew from 60 to 102 prompts with
fixtures (outlets, switches, vents, pipes, columns) and things of offices, kitchens and bathrooms. Some were chosen
after seeing what SAM missed on Replica with its own class names (ceiling lights asked for as "lamp", wall plugs,
panels), so the list isn't blind to Replica's kinds of objects, though no prompt is a Replica class name chosen for one
scene.
"""

from pathlib import Path

INDOOR = [
    "air vent",
    "armchair",
    "backpack",
    "bag",
    "basket",
    "bathtub",
    "bed",
    "bench",
    "blanket",
    "blinds",
    "book",
    "bookshelf",
    "bottle",
    "bowl",
    "box",
    "bulletin board",
    "cabinet",
    "cable",
    "camera",
    "candle",
    "ceiling light",
    "chair",
    "clock",
    "clothes",
    "coat rack",
    "coffee maker",
    "column",
    "computer mouse",
    "counter",
    "cup",
    "curtain",
    "cushion",
    "desk",
    "desk organizer",
    "dishwasher",
    "door",
    "drawer",
    "fan",
    "faucet",
    "fire extinguisher",
    "flower pot",
    "headphones",
    "jar",
    "kettle",
    "keyboard",
    "lamp",
    "laptop",
    "light switch",
    "microwave",
    "mirror",
    "monitor",
    "mug",
    "ottoman",
    "oven",
    "pan",
    "paper towel",
    "partition",
    "phone",
    "picture",
    "pillow",
    "pipe",
    "plant",
    "plate",
    "poster",
    "power outlet",
    "power strip",
    "printer",
    "projector",
    "radiator",
    "refrigerator",
    "remote control",
    "rug",
    "sculpture",
    "shelf",
    "shoe",
    "shower",
    "sink",
    "smoke detector",
    "soap dispenser",
    "sofa",
    "speaker",
    "stool",
    "storage bin",
    "stove",
    "table",
    "tablet",
    "television",
    "thermostat",
    "tissue box",
    "toaster",
    "toilet",
    "toilet paper",
    "towel",
    "toy",
    "trash can",
    "tray",
    "tv stand",
    "umbrella",
    "vase",
    "wardrobe",
    "whiteboard",
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
