import unittest
from unittest import mock

import numpy as np

from graph import shape
from utils.schema import Boxes, Masks, Run, encode_rle

SIZE = 400
K = np.array([[100.0, 0, SIZE / 2], [0, 100.0, SIZE / 2], [0, 0, 1]])  # 127 degrees across: sees all of the streak
CUBE = 0.25  # half the side of a cube at the origin
DISTANCE = 3.0  # of both cameras from the origin
SMEAR = 2.0  # metres the front view's depth runs on behind the cube's front face, from its top row to its bottom one


def camera(centre, x_axis, y_axis):
    """Camera-to-world (4, 4) of an OpenCV camera at centre looking along x_axis x y_axis."""
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x_axis, y_axis, np.cross(x_axis, y_axis)], axis=1)
    c2w[:3, 3] = centre
    return c2w


VIEWS = [  # from in front (looking along +z) and from the side (looking along -x)
    camera([0, 0, -DISTANCE], [1, 0, 0], [0, 1, 0]),
    camera([DISTANCE, 0, 0], [0, 0, 1], [0, 1, 0]),
]


class ObjectBoxTests(unittest.TestCase):
    def test_a_view_whose_depth_smears_the_object_back_is_cut_to_what_the_other_view_sees(self):
        # Both views see the cube as a square: its near face, DISTANCE - CUBE away
        near = DISTANCE - CUBE
        offset = np.abs(np.arange(SIZE) - SIZE / 2)
        square = (offset[:, None] <= K[0, 0] * CUBE / near) & (offset[None, :] <= K[0, 0] * CUBE / near)
        top, bottom = np.flatnonzero(square.any(axis=1))[[0, -1]]
        ramp = np.clip((np.arange(SIZE) - top) / (bottom - top), 0, 1)[:, None]
        depth = [
            np.where(square, near + SMEAR * ramp, 10.0).astype(np.float32),  # wrong: 2 m deep at the bottom
            np.where(square, near, 10.0).astype(np.float32),
        ]
        speck = np.zeros((SIZE, SIZE), bool)
        speck[:2, :2] = True  # too few pixels to lift: its sample from the boxes stage is used
        points = np.full((4, 256, 3), np.nan, np.float16)  # rows 0 and 1 are lifted here, not taken from the sample
        points[3, :4] = [[1, 1, 1], [1.5, 1, 1], [1, 1.5, 1], [1, 1, 1.5]]
        box_min = np.array([[-0.4, -0.4, -CUBE], [-CUBE] * 3, [5, 5, 5], [1, 1, 1]], np.float32)
        box_max = np.array([[0.4, 0.4, CUBE + SMEAR], [CUBE] * 3, [5.5, 5.5, 5.5], [1.5, 1.5, 1.5]], np.float32)
        boxes = Boxes(
            box_min,
            box_max,
            np.array([256, 256, 256, 4], np.int32),
            points,
            np.array([5.0, 5.0, 1.2, 5.0], np.float32),  # the third mask's depth is not trusted: no reliable view
        )
        masks = Masks(
            np.array([0, 1, 1, 1], np.int32),
            np.zeros(4, np.int16),
            np.array([1, 1, 2, 3], np.int32),
            np.ones(4, np.float32),
            encode_rle(np.stack([square, square, square, speck])),
        )
        run = Run("replica/room_0/00", 1, "sam3", "vggt")
        get = lambda frame: (depth[frame], K, VIEWS[frame])  # noqa: E731
        with mock.patch.object(shape, "view_source", return_value=get):
            lo, hi = shape.object_boxes(run, masks, boxes, np.array([0, 0, 2, 4], np.int32))
        # Within 5 cm: the front view's rays spread out, so its points just inside the side view's square reach 0.29
        np.testing.assert_allclose(lo[0], -CUBE, atol=0.05)
        np.testing.assert_allclose(hi[0], CUBE, atol=0.05)  # not 2 m deep
        self.assertTrue(np.isnan(lo[1]).all())  # no mask has node 1
        np.testing.assert_array_equal(lo[2], boxes.box_min[2])  # the most confident mask's box
        np.testing.assert_array_equal(hi[2], boxes.box_max[2])
        np.testing.assert_allclose(lo[4], [1, 1, 1], atol=0.01)
        np.testing.assert_allclose(hi[4], [1.5, 1.5, 1.5], atol=0.01)


if __name__ == "__main__":
    unittest.main()
