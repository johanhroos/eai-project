"""Keep smaller ROCm GPUs within their memory budget without slowing larger GPUs."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from models import sam


class MemorySettingsTests(unittest.TestCase):
    def settings(self, hip, gib, state=None, batch=None):
        with (
            patch.object(sam.torch.version, "hip", hip),
            patch.object(
                sam.torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=gib * 2**30)
            ),
            patch.object(sam, "STATE_DEVICE", state),
            patch.object(sam, "DETECT_BATCH", batch),
        ):
            return sam.memory_settings()

    def test_small_rocm_gpu_offloads_tracking_and_limits_detection(self):
        for gib in (8, 16):
            with self.subTest(gib=gib):
                self.assertEqual(self.settings("7.2", gib), ("cpu", 1))

    def test_larger_rocm_gpu_keeps_gpu_tracking_and_batching(self):
        self.assertEqual(self.settings("7.2", 24), ("cuda", 16))

    def test_nvidia_defaults_stay_the_same(self):
        self.assertEqual(self.settings(None, 16), ("cuda", 16))

    def test_explicit_settings_override_the_automatic_defaults(self):
        self.assertEqual(self.settings("7.2", 16, state="cuda", batch=4), ("cuda", 4))


class MemoryEncodingTests(unittest.TestCase):
    def test_chunks_preserve_object_order_scores_and_shared_or_batched_vision(self):
        masks = torch.arange(37, dtype=torch.float32).reshape(37, 1, 1, 1)
        scores = torch.arange(37, dtype=torch.float32).reshape(37, 1)

        def reference(current_vision_feats, pred_masks_high_res, object_score_logits, is_mask_from_pts):
            mask = pred_masks_high_res.flatten().reshape(1, -1, 1)
            score = object_score_logits.reshape(1, -1, 1)
            out = current_vision_feats + mask + score
            return out, out + int(is_mask_from_pts)

        for vision_count in (1, 37):
            for from_points in (False, True):
                with self.subTest(vision_count=vision_count, from_points=from_points):
                    vision = torch.arange(3 * vision_count * 2, dtype=torch.float32).reshape(3, vision_count, 2)
                    encode = Mock(side_effect=reference)
                    actual = sam.chunked_memory_encoding(encode)(vision, masks, scores, from_points)
                    expected = reference(vision, masks, scores, from_points)
                    calls = [len(call.kwargs["pred_masks_high_res"]) for call in encode.call_args_list]
                    self.assertEqual(calls, [16, 16, 5])
                    for got, want in zip(actual, expected, strict=True):
                        torch.testing.assert_close(got, want)


class MaskIoUTests(unittest.TestCase):
    def test_matches_binary_intersections_exactly_under_autocast(self):
        generator = torch.Generator().manual_seed(13)
        pred = torch.rand(7, 32, 33, generator=generator) > 0.4
        gt = torch.rand(11, 32, 33, generator=generator) > 0.6
        pred[0] = False
        gt[0] = False
        pred[1] = True
        gt[1] = True
        intersection = (pred[:, None] & gt[None]).sum((-1, -2)).float()
        union = (pred[:, None] | gt[None]).sum((-1, -2)).float()
        expected = intersection / union.clamp_min(1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = sam.mask_iou(pred, gt)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.float32)

    def test_empty_mask_batches_keep_their_pairwise_shape(self):
        for n, m in ((0, 5), (5, 0), (0, 0)):
            with self.subTest(n=n, m=m):
                actual = sam.mask_iou(torch.zeros(n, 3, 3, dtype=torch.bool), torch.zeros(m, 3, 3, dtype=torch.bool))
                self.assertEqual(actual.shape, (n, m))


if __name__ == "__main__":
    unittest.main()
