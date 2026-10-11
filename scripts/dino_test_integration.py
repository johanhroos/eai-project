"""Minimal example of integrating DINOv3 object descriptors."""

from models.dino import DINOEncoder
from utils import replica


def main():
    dino = DINOEncoder("facebook/dinov3-vitl16-pretrain-lvd1689m")

    rgb, _, _ = replica.load_frame("room_0", "00", 0)
    bboxes = [(100, 100, 250, 300), (350, 150, 500, 400)]

    descriptors, valid = dino.encode_objects(rgb, bboxes, inclusive=True)

    print("Descriptors:", descriptors.shape)
    print("Valid:", valid.tolist())


if __name__ == "__main__":
    main()
