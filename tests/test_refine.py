import unittest
import warnings
from unittest import mock

import numpy as np
from scipy.ndimage import binary_dilation

import evaluate
from graph import query, refine
from utils.replica import ReplicaObject
from utils.schema import Boxes, Events, Features, Masks, Run, TextAligned, encode_rle

RNG = np.random.default_rng(0)
SIZE = 100  # pixels on each side of the images


def cube(lo, side=0.3, n=256):
    """(n, 3) points spread through the cube with corner lo."""
    return np.asarray(lo, np.float32) + RNG.uniform(0, side, (n, 3)).astype(np.float32)


def stages(rows, pixels=None):
    """Masks, Boxes and Features from rows of (frame, prompt, SAM ID, points (k, 3) or None, DINO (2,), conf).

    pixels gives each mask's area (a square in the corner of the image), 400 by default.
    """
    n = len(rows)
    image = np.zeros((n, SIZE, SIZE), bool)
    for i, area in enumerate(pixels or [400] * n):
        side = int(np.sqrt(area))
        image[i, :side, :side] = True
    points = np.full((n, 256, 3), np.nan, np.float16)
    for i, row in enumerate(rows):
        if row[3] is not None:
            points[i, : len(row[3])] = row[3]
    masks = Masks(
        np.array([r[0] for r in rows], np.int32),
        np.array([r[1] for r in rows], np.int16),
        np.array([r[2] for r in rows], np.int32),
        np.ones(n, np.float32),
        encode_rle(image),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # masks without points
        lo, hi = np.nanmin(points, axis=1).astype(np.float32), np.nanmax(points, axis=1).astype(np.float32)
    has = np.array([r[3] is not None for r in rows])
    boxes = Boxes(lo, hi, np.where(has, 256, 0).astype(np.int32), points, np.array([r[5] for r in rows], np.float32))
    features = Features(np.array([r[4] for r in rows], np.float16))
    return masks, boxes, features


LOOK = (1.0, 0.0)  # a DINO descriptor
OTHER_LOOK = (0.0, 1.0)


class SplitTests(unittest.TestCase):
    def test_a_node_over_two_clusters_is_split_and_a_mask_over_both_goes_with_its_larger_share(self):
        a, b = (0, 0, 0), (1.0, 0, 0)
        between = np.c_[RNG.uniform(0.3, 1.0, 16), RNG.uniform(0, 0.3, (16, 2))].astype(np.float32)
        rows = [
            (0, 0, 1, cube(a), LOOK, 5.0),
            (1, 0, 1, cube(a), LOOK, 5.0),
            (2, 0, 1, np.r_[cube(a, n=150), cube(b, n=90), between], LOOK, 5.0),  # covers both, more of a
            (3, 0, 2, cube(b), LOOK, 5.0),
            (4, 0, 2, cube(b), LOOK, 5.0),
            (5, 0, 2, cube(b), LOOK, 1.0),  # depth not trusted: follows its track
        ]
        masks, boxes, _ = stages(rows)
        node, records = refine.split(masks, boxes, np.zeros(6, np.int32))
        np.testing.assert_array_equal(node, [0, 0, 0, 1, 1, 1])
        self.assertEqual(records, [{"node": 0, "parts": [0, 1], "masks": [3, 3]}])

    def test_one_object_seen_in_parts_stays_whole(self):
        plane = lambda x0, x1: np.c_[RNG.uniform(x0, x1, (256, 2)), np.zeros(256)].astype(np.float32)  # noqa: E731
        rows = [(f, 0, 1, plane(x0, x0 + 1.2), LOOK, 5.0) for f, x0 in enumerate((0.0, 0.4, 0.8, 1.2))]
        masks, boxes, _ = stages(rows)
        node, records = refine.split(masks, boxes, np.zeros(4, np.int32))
        np.testing.assert_array_equal(node, 0)
        self.assertEqual(records, [])


class MergeTests(unittest.TestCase):
    def merge(self, part, part_frames=(10, 11)):
        """Merge a node of masks (prompt, points, DINO) in part_frames with a whole: 4 masks of a 1 m cube in 0-3."""
        rows = [(f, 0, 1, cube((0, 0, 0), side=1.0), LOOK, 5.0) for f in range(4)]
        rows += [(f, prompt, 2, points, look, 5.0) for f, (prompt, points, look) in zip(part_frames, part, strict=True)]
        masks, boxes, features = stages(rows)
        node = np.array([0] * 4 + [1] * len(part), np.int32)
        lo = np.array([[0, 0, 0], [0.25, 0.25, 0.25]], np.float32)
        hi = np.array([[1, 1, 1], [0.75, 0.75, 0.75]], np.float32)
        return refine.merge(masks, boxes, features, node, lo, hi, refine.labels(masks, node))

    def test_a_part_inside_a_whole_of_its_label_and_look_joins_it(self):
        node, records = self.merge([(0, cube((0.25, 0.25, 0.25), 0.5), LOOK)] * 2)
        np.testing.assert_array_equal(node, 0)
        self.assertEqual([(r["kept"], r["merged"]) for r in records], [(0, 1)])

    def test_another_label_another_look_or_a_frame_together_keeps_them_apart(self):
        part = cube((0.25, 0.25, 0.25), 0.5)
        for case in (
            {"part": [(1, part, LOOK)] * 2},
            {"part": [(0, part, OTHER_LOOK)] * 2},
            {"part": [(0, part, LOOK)] * 2, "part_frames": (3, 11)},  # seen with the whole in frame 3
        ):
            node, records = self.merge(**case)
            np.testing.assert_array_equal(node, [0] * 4 + [1] * 2)
            self.assertEqual(records, [])

    def test_a_part_off_the_wholes_points_or_seen_more_stays(self):
        node, _ = self.merge([(0, cube((5, 5, 5), 0.5), LOOK)] * 2)  # points off the whole, where its box says
        np.testing.assert_array_equal(node, [0] * 4 + [1] * 2)
        # Seen in more masks than the whole: then the whole must lie inside it, and the 1 m box doesn't
        node, _ = self.merge([(0, cube((0.25, 0.25, 0.25), 0.5), LOOK)] * 5, part_frames=range(10, 15))
        np.testing.assert_array_equal(node, [0] * 4 + [1] * 5)


class LabelTests(unittest.TestCase):
    def test_siglip_decides_between_close_votes(self):
        rows = [(f, prompt, 1, cube((0, 0, 0)), LOOK, 5.0) for f, prompt in enumerate((0, 0, 0, 1, 1))]
        masks, _, _ = stages(rows)
        node = np.zeros(5, np.int32)
        emb = TextAligned(np.tile(np.array([0.6, 0.8], np.float16), (5, 1)))  # nearer prompt 1's text
        text = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
        self.assertEqual(refine.labels(masks, node), {0: 0})  # 3 votes to 2
        self.assertEqual(refine.labels(masks, node, emb, text), {0: 1})
        # SigLIP only chooses among the prompts SAM gave the node, however well another one fits
        masks = Masks(masks.frame, np.zeros(5, np.int16), masks.sam_id, masks.score, masks.rle)
        self.assertEqual(refine.labels(masks, node, TextAligned(np.tile(np.float16([0, 1]), (5, 1))), text), {0: 0})


class ConsensusTests(unittest.TestCase):
    def test_a_node_sees_its_points_in_its_mask_but_not_those_hidden_behind_the_wall(self):
        image = np.zeros((1, SIZE, SIZE), bool)
        image[0, 40:60, 40:60] = True
        masks = Masks(*(np.zeros(1, t) for t in (np.int32, np.int16, np.int32, np.float32)), encode_rle(image))
        K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
        depth = np.full((SIZE, SIZE), 2.0, np.float32)  # a wall 2 m in front of the camera
        points = {
            0: np.c_[RNG.uniform(-0.15, 0.15, (64, 2)), np.full(64, 2.0)].astype(np.float32),  # on the wall, masked
            1: np.c_[RNG.uniform(-0.15, 0.15, (64, 2)), np.full(64, 4.0)].astype(np.float32),  # behind it
            2: np.c_[RNG.uniform(0.6, 0.9, (64, 2)), np.full(64, 2.0)].astype(np.float32),  # on it, outside the mask
        }
        run = Run("phone/test", 1, "sam3", "vggt")
        with mock.patch.object(refine.shape, "view_source", return_value=lambda f: (depth, K, np.eye(4))):
            seen = refine.sightings(run, masks, points)
        self.assertEqual(seen, {0: {0: {0}, 2: set()}})

    def view(self, frames, together=(), labels=(0, 0)):
        """Nodes 0 and 1 in view in every frame, held by one mask in frames, each by its own mask in together."""
        rows, seen = [], {}
        for f in range(6):
            if f in frames:
                rows.append((f, labels[0], 1))
                seen[f] = {0: {len(rows) - 1}, 1: {len(rows) - 1}}
            elif f in together:
                rows += [(f, labels[0], 1), (f, labels[1], 2)]
                seen[f] = {0: {len(rows) - 2}, 1: {len(rows) - 1}}
            else:
                rows.append((f, labels[0], 1))
                seen[f] = {0: {len(rows) - 1}, 1: set()}
        node = np.array([0 if r[2] == 1 else 1 for r in rows], np.int32)
        node[[i for i, r in enumerate(rows) if r[0] in frames][1:]] = 1  # node 1 owns some of the shared masks
        masks = Masks(
            np.array([r[0] for r in rows], np.int32),
            np.array([r[1] for r in rows], np.int16),
            np.array([r[2] for r in rows], np.int32),
            np.ones(len(rows), np.float32),
            encode_rle(np.zeros((len(rows), 4, 4), bool)),
        )
        return refine.consensus_merge(masks, node, {0: labels[0], 1: labels[1]}, seen)

    def test_two_nodes_one_mask_holds_wherever_both_are_seen_merge(self):
        node, records = self.view(frames=range(6))
        self.assertTrue((node == node[0]).all())
        self.assertEqual(len(records), 1)

    def test_too_few_frames_agreeing_two_masks_or_two_labels_keep_them_apart(self):
        for case in (
            {"frames": range(3)},  # the other 3 frames see both and hold only node 0: consensus 0.5
            {"frames": range(2), "together": range(2, 6)},
            {"frames": range(6), "labels": (0, 1)},
        ):
            node, records = self.view(**case)
            self.assertEqual(records, [])
            self.assertEqual(len(np.unique(node)), 2)


class SeenApartTests(unittest.TestCase):
    def test_two_nodes_in_a_few_frames_together_are_a_slip_and_in_many_two_objects(self):
        self.assertFalse(refine.seen_apart(set(range(10)), set(range(10, 20)), 0.2))
        self.assertFalse(refine.seen_apart(set(range(10)), {9, 10, 11, 12, 13}, 0.2))  # 1 of 5 frames
        self.assertTrue(refine.seen_apart(set(range(10)), {8, 9, 10, 11, 12}, 0.2))  # 2 of 5
        self.assertTrue(refine.seen_apart(set(range(10)), {9, 10}, 0.0))


class DuplicateTests(unittest.TestCase):
    def merge(self, inner, labels=(0, 0), boxes3d=None):
        """Node 0 a 40 px square in frames 0-2, node 1 inner (a mask) in the same frames, both of labels."""
        outer = np.zeros((SIZE, SIZE), bool)
        outer[20:60, 20:60] = True
        image = np.stack([outer, inner] * 3)
        masks = Masks(
            np.repeat(np.arange(3, dtype=np.int32), 2),
            np.tile(np.int16(labels), 3),
            np.tile(np.int32([1, 2]), 3),
            np.ones(6, np.float32),
            encode_rle(image),
        )
        node = np.tile(np.int32([0, 1]), 3)
        return refine.duplicate_merge(masks, node, {0: labels[0], 1: labels[1]}, boxes3d=boxes3d, nest=0.5)

    def test_a_mask_inside_another_of_its_label_in_the_frames_they_share_is_the_same_object(self):
        seat = np.zeros((SIZE, SIZE), bool)
        seat[40:60, 20:60] = True  # the chair's seat, inside the chair
        node, records = self.merge(seat)
        np.testing.assert_array_equal(node, 0)
        self.assertEqual([(r["kept"], r["merged"], r["inside"]) for r in records], [(0, 1, 1.0)])

    def test_another_label_half_inside_or_boxes_apart_stay_two(self):
        seat, half = np.zeros((SIZE, SIZE), bool), np.zeros((SIZE, SIZE), bool)
        seat[40:60, 20:60] = True
        half[40:80, 20:60] = True  # half of it outside
        apart = (np.float32([[0, 0, 0], [5, 5, 5]]), np.float32([[1, 1, 1], [6, 6, 6]]))
        for case in ({"inner": seat, "labels": (0, 1)}, {"inner": half}, {"inner": seat, "boxes3d": apart}):
            node, records = self.merge(**case)
            np.testing.assert_array_equal(node, np.tile([0, 1], 3))
            self.assertEqual(records, [])


class DropTests(unittest.TestCase):
    def test_nodes_of_few_masks_and_of_only_small_far_ones_are_dropped(self):
        far, near = cube((6, 0, 0)), cube((1, 0, 0))
        rows = [(f, 0, 1, far, LOOK, 5.0) for f in range(3)]  # node 0: small and far
        rows += [(f, 0, 2, p, LOOK, 5.0) for f, p in enumerate((far, far, near))]  # node 1: once close
        rows += [(f, 0, 3, near, LOOK, 5.0) for f in range(2)]  # node 2: two masks
        rows += [(f, 0, 4, None, LOOK, 5.0) for f in range(3)]  # node 3: small, no depth
        masks, boxes, _ = stages(rows, pixels=[9] * len(rows))
        node = np.repeat(np.arange(4), [3, 3, 2, 3]).astype(np.int32)
        run = Run("phone/test", 1, "sam3", "vggt")
        trusted = {n: {"coverage": 1.0, "views": 99, "arc": 360.0} for n in range(4)}
        with (
            mock.patch.object(refine, "camera_centres", return_value={f: np.zeros(3) for f in range(3)}),
            mock.patch.object(refine, "quality", return_value=trusted),
        ):
            dropped, records = refine.drop(run, masks, boxes, node, min_masks=3, small=0.002, far=4.0)
            trusted[1] = {"coverage": 0.3, "views": 99, "arc": 5.0}
            _, untrusted = refine.drop(run, masks, boxes, node, min_masks=3, small=0.002, far=4.0)
        np.testing.assert_array_equal(dropped, [-1] * 3 + [1] * 3 + [-1] * 5)
        reasons = [(0, ["small and far"]), (2, ["few masks"]), (3, ["small and far"])]
        self.assertEqual([(r["node"], r["reasons"]) for r in records], reasons)
        self.assertEqual(untrusted[1]["reasons"], ["inconsistent", "narrow views"])

    def test_quality_sees_another_objects_points_and_the_views_arc(self):
        K = np.array([[50.0, 0, 50], [0, 50.0, 50], [0, 0, 1]])
        depth = np.full((SIZE, SIZE), 10.0, np.float32)  # nothing in front of the objects

        def camera(azimuth):  # 2 m from the origin, level, looking at it
            centre = 2 * np.array([np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth)), 0.0])
            forward = -centre / 2
            right = np.cross(forward, [0, 0, 1.0])
            c2w = np.eye(4)
            c2w[:3, :3] = np.stack([right, np.cross(forward, right), forward], axis=1)  # x right, y down, z ahead
            c2w[:3, 3] = centre
            return c2w

        cameras = {f: camera(a) for f, a in enumerate((0, 45, 90, 0, 0, 0))}
        thing = cube((-0.1, -0.1, -0.1), side=0.2)

        def mask_of(points, f):
            u, v, seen = refine.project(points, depth, K, cameras[f])
            image = np.zeros((SIZE, SIZE), bool)
            image[v[seen], u[seen]] = True
            return binary_dilation(image, iterations=2)

        image = [mask_of(thing, f) for f in range(3)]  # node 0: the thing, from 0, 45 and 90 degrees
        image += [~mask_of(thing, f) for f in range(3, 6)]  # node 1: its points, outside its masks, all from 0 degrees
        masks = Masks(
            np.arange(6, dtype=np.int32),
            np.zeros(6, np.int16),
            np.repeat(np.int32([1, 2]), 3),
            np.ones(6, np.float32),
            encode_rle(np.stack(image)),
        )
        points = np.full((6, 256, 3), np.nan, np.float16)
        points[:] = thing[:256]
        boxes = Boxes(*(np.zeros((6, 3), np.float32) for _ in range(2)), np.full(6, 256, np.int32), points)
        run = Run("phone/test", 1, "sam3", "vggt")
        with mock.patch.object(refine.shape, "view_source", return_value=lambda f: (depth, K, cameras[f])):
            q = refine.quality(run, masks, boxes, np.repeat(np.int32([0, 1]), 3))
        self.assertGreater(q[0]["coverage"], 0.9)
        self.assertEqual(q[0]["views"], 3)
        self.assertAlmostEqual(q[0]["arc"], 90.0, delta=2.0)  # to the median of its points, near the cube's centre
        self.assertLess(q[1]["coverage"], 0.1)
        self.assertEqual(q[1]["arc"], 0.0)


class LeftOutTests(unittest.TestCase):
    def test_masks_of_no_node_are_no_prediction(self):
        side = 40
        image = np.zeros((3, side, side), bool)
        image[[0, 2], :30, :30] = True
        image[1, 32:38, 32:38] = True  # something no GT object is
        masks = Masks(
            np.array([0, 0, 1], np.int32),
            np.zeros(3, np.int16),
            np.array([1, 2, 1], np.int32),
            np.ones(3, np.float32),
            encode_rle(image),
        )
        gt = Masks(
            np.array([0, 1], np.int32),
            np.zeros(2, np.int16),
            np.array([5, 5], np.int32),
            np.ones(2, np.float32),
            encode_rle(image[[0, 2]]),
        )
        events = Events(*(np.zeros(0, t) for t in (np.int32, np.int32, np.int8, np.int32)))
        objects = {5: ReplicaObject(5, "chair")}
        with mock.patch.object(evaluate.lift3d, "MIN_PIXELS", 100):
            left_out = evaluate.evaluate(masks, np.array([0, -1, 0]), gt, events, objects)
            own_node = evaluate.evaluate(masks, np.array([0, 1, 0]), gt, events, objects)
        self.assertEqual((left_out["ids"], left_out["idp"], left_out["idf1"]), (1, 1.0, 1.0))
        self.assertEqual((own_node["ids"], own_node["idp"]), (2, 2 / 3))
        emb = TextAligned(np.array([[1, 0], [0, 1], [1, 0]], np.float16))
        ids, embeddings = query.object_embeddings(masks, emb, np.array([0, -1, 0]))
        np.testing.assert_array_equal(ids, [0])
        np.testing.assert_allclose(embeddings, [[1, 0]])


class VocabTextTests(unittest.TestCase):
    def test_the_prompts_text_is_encoded_once_per_vocabulary(self):
        import tempfile
        from pathlib import Path

        import torch

        from graph import pipeline
        from models import siglip

        calls = []

        class Encoder:
            def encode_text(self, texts):
                calls.append(list(texts))
                return torch.ones(len(texts), 2)

        with tempfile.TemporaryDirectory() as temporary:
            run = Run("phone/test", 1, "sam3", "vggt")
            Path(temporary, "siglip").mkdir()
            with (
                mock.patch.object(Run, "stage_dir", lambda self, stage: Path(temporary) / stage),
                mock.patch.object(siglip, "load_encoder", return_value=Encoder()),
                mock.patch.object(torch.cuda, "is_available", return_value=True),
            ):
                first = pipeline.vocab_text(run, ["chair", "table"])
                again = pipeline.vocab_text(run, ["chair", "table"])
                other = pipeline.vocab_text(run, ["chair"])
            with (
                mock.patch.object(Run, "stage_dir", lambda self, stage: Path(temporary) / stage),
                mock.patch.object(torch.cuda, "is_available", return_value=False),
            ):
                cached = pipeline.vocab_text(run, ["chair"])
                uncached = pipeline.vocab_text(run, ["lamp"])  # no GPU to encode it: SAM's votes alone
        self.assertEqual(calls, [["chair", "table"], ["chair"]])
        np.testing.assert_array_equal(first, again)
        self.assertEqual(other.shape, (1, 2))
        np.testing.assert_array_equal(cached, other)
        self.assertIsNone(uncached)


if __name__ == "__main__":
    unittest.main()
