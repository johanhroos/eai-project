"""Reject invalid model predictions before NumPy camera alignment or cache writes."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from data.stages import Run
from models import vggt


class PredictionValidationTests(unittest.TestCase):
    def test_nonfinite_predictions_fail_before_camera_decoding(self):
        run = Run("phone/example", 20, "sam3", "vggt")
        images = Mock()
        images.to.return_value = images
        for name in ("pose_enc", "depth", "depth_conf"):
            for bad in (float("nan"), float("inf")):
                with self.subTest(name=name, bad=bad):
                    predictions = {key: torch.ones(1) for key in ("pose_enc", "depth", "depth_conf")}
                    predictions[name][0] = bad
                    with (
                        patch.object(vggt.footage, "frame_numbers", return_value=np.array([0, 20])),
                        patch.object(vggt.footage, "image_path", return_value=Path("example.jpg")),
                        patch.object(vggt, "load_and_preprocess_images", return_value=images),
                        patch.object(vggt, "encoding_to_camera") as decode,
                        self.assertRaisesRegex(RuntimeError, f"non-finite {name}"),
                    ):
                        vggt.reconstruct(run, Mock(return_value=predictions))
                    decode.assert_not_called()

    def test_nonfinite_cameras_report_the_prediction_problem(self):
        for bad in (np.nan, np.inf):
            cameras = np.tile(np.eye(4)[:3], (3, 1, 1))
            cameras[1, 0, 0] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "non-finite camera poses"):
                vggt.gravity_up(cameras)

    def test_valid_stationary_cameras_use_their_up_axis(self):
        cameras = np.tile(np.eye(4)[:3], (3, 1, 1))
        np.testing.assert_allclose(vggt.gravity_up(cameras), [0, -1, 0])


if __name__ == "__main__":
    unittest.main()
