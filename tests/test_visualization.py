import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
from PIL import Image

from graph.visualize import overlay_png
from utils.schema import Masks, Run, encode_rle


class FrameDisplayTests(unittest.TestCase):
    def test_frame_png_has_overlay_and_ids_without_plotly_display(self):
        rgb = np.full((64, 64, 3), 100, np.uint8)
        mask = np.zeros((1, 64, 64), bool)
        mask[:, 12:48, 12:48] = True
        masks = Masks(
            np.array([0], np.int32),
            np.array([0], np.int16),
            np.array([99], np.int32),
            np.ones(1, np.float32),
            encode_rle(mask),
        )
        run = Run("replica/room_0/00", 20, "sam3", "gt")
        with (
            patch("graph.visualize.footage.load_rgb", return_value=rgb),
            patch("plotly.graph_objects.Figure.show") as show,
        ):
            png = overlay_png(run, masks, np.array([7]), 0)
            self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
            decoded = np.asarray(Image.open(BytesIO(png)))
            self.assertEqual(decoded.shape, rgb.shape)
            self.assertFalse(np.array_equal(decoded, rgb))
            # Selecting an object absent from the frame leaves RGB unchanged.
            absent = overlay_png(run, masks, np.array([7]), 0, selected=8)
            np.testing.assert_array_equal(np.asarray(Image.open(BytesIO(absent))), rgb)
            show.assert_not_called()


if __name__ == "__main__":
    unittest.main()
