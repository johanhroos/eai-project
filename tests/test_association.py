import unittest

import numpy as np

from graph.association import AssociationConfig, ObjectMemory, Observation
from graph.pipeline import replay
from utils.lift3d import Box3D
from utils.schema import Boxes, Features, Masks, TextAligned, encode_rle


def observation(descriptor=(1, 0), x=0, label="chair", sam_id=1):
    return Observation(
        np.array(descriptor, np.float32),
        Box3D(np.array([x, 0.0, 0.0]), np.array([x + 1, 1.0, 1.0]), 100),
        label,
        sam_id,
        np.array(descriptor, np.float32),
    )


class AssociationTests(unittest.TestCase):
    def test_cached_stage_replay_preserves_row_order(self):
        masks = Masks(
            np.array([10, 0], np.int32),
            np.zeros(2, np.int16),
            np.array([99, 1], np.int32),
            np.ones(2, np.float32),
            encode_rle(np.ones((2, 4, 4), bool)),
        )
        boxes = Boxes(np.zeros((2, 3), np.float32), np.ones((2, 3), np.float32), np.ones(2, np.int32))
        features = Features(np.array([[1, 0], [1, 0]], np.float16))
        result = replay(masks, boxes, features, TextAligned(features.dino.copy()), ["chair"])
        np.testing.assert_array_equal(result.graph.node, [0, 0])
        self.assertEqual(result.memory.nodes[0].first_seen, 0)
        self.assertEqual(result.memory.nodes[0].last_seen, 10)

    def test_revisit_with_new_sam_id(self):
        memory = ObjectMemory()
        first = memory.associate_frame(0, [observation()])[0]
        memory.associate_frame(1, [])
        self.assertEqual(len(memory.nodes), 1)
        revisited = memory.associate_frame(20, [observation(sam_id=99)])[0]
        self.assertEqual(first, revisited)
        self.assertEqual(memory.nodes[first].observation_count, 2)

    def test_distinct_objects_and_one_to_one(self):
        memory = ObjectMemory()
        ids = memory.associate_frame(0, [observation(), observation(x=4)])
        self.assertNotEqual(*ids)
        np.testing.assert_array_equal(memory.associate_frame(3, [observation(x=4), observation()]), ids[::-1])
        # Two observations cannot update the same existing node in one frame.
        result = memory.associate_frame(4, [observation(), observation()])
        self.assertEqual(len(set(result)), 2)

    def test_appearance_and_label_separate_objects(self):
        memory = ObjectMemory(AssociationConfig(require_same_label=True))
        original = memory.associate_frame(0, [observation()])[0]
        other = memory.associate_frame(1, [observation(descriptor=(0, 1))])[0]
        self.assertNotEqual(original, other)
        different_label = memory.associate_frame(2, [observation(label="table")])[0]
        self.assertNotIn(different_label, [original, other])

    def test_max_descriptor_score_and_bounded_bank(self):
        memory = ObjectMemory(AssociationConfig(threshold=0.39, max_descriptors=2))
        object_id = memory.associate_frame(0, [observation()])[0]
        self.assertEqual(memory.associate_frame(1, [observation(descriptor=(0, 1))])[0], object_id)
        self.assertAlmostEqual(memory.score(observation(descriptor=(0, 1)), memory.nodes[object_id]), 1)
        memory.associate_frame(2, [observation(descriptor=(1, 1))])
        self.assertEqual(len(memory.nodes[object_id].descriptors), 2)

    def test_invalid_inputs_do_not_poison_memory(self):
        memory = ObjectMemory(AssociationConfig(threshold=0.3))
        object_id = memory.associate_frame(0, [observation()])[0]
        invalid_descriptor = observation(descriptor=(np.nan, 0))
        self.assertEqual(memory.associate_frame(1, [invalid_descriptor])[0], object_id)
        self.assertEqual(len(memory.nodes[object_id].descriptors), 1)
        invalid = Observation(np.zeros(2), Box3D(np.full(3, np.nan), np.ones(3), 0), "chair")
        new_id = memory.associate_frame(2, [invalid])[0]
        self.assertNotEqual(object_id, new_id)
        self.assertIsNone(memory.nodes[new_id].box)
        self.assertFalse(memory.nodes[new_id].descriptors)
        self.assertTrue(np.isfinite(memory.score(observation(descriptor=(1, 0, 0)), memory.nodes[object_id])))

    def test_partial_geometry_does_not_shrink(self):
        memory = ObjectMemory(AssociationConfig(threshold=0.5))
        object_id = memory.associate_frame(0, [observation()])[0]
        partial = observation()
        partial.box.max[:] = 0.5
        self.assertEqual(memory.associate_frame(1, [partial])[0], object_id)
        np.testing.assert_array_equal(memory.nodes[object_id].box.max, [1, 1, 1])

    def test_box_is_union_of_views(self):
        memory = ObjectMemory()
        object_id = memory.associate_frame(0, [observation()])[0]
        # A view that also sees the part behind an occluder: the box grows to cover both views at once
        wider = observation(x=0.5)
        self.assertEqual(memory.associate_frame(1, [wider])[0], object_id)
        np.testing.assert_array_equal(memory.nodes[object_id].box.min, [0, 0, 0])
        np.testing.assert_array_equal(memory.nodes[object_id].box.max, [1.5, 1, 1])

    def test_partial_view_inside_the_box_counts_as_the_same_place(self):
        memory = ObjectMemory()
        object_id = memory.associate_frame(0, [observation()])[0]
        memory.associate_frame(1, [observation(x=0.5)])  # the box is now 1.5 m long
        part = observation(x=0.2)
        part.box = Box3D(np.array([0.2, 0.2, 0.2]), np.array([0.4, 0.4, 0.4]), 100)  # a small part, inside the box
        parts = memory.components(part, memory.nodes[object_id])
        self.assertEqual(parts["inside"], 1)
        self.assertEqual(memory.associate_frame(2, [part])[0], object_id)

    def test_strict_threshold_and_frame_order(self):
        memory = ObjectMemory(AssociationConfig(threshold=1))
        first = memory.associate_frame(0, [observation()])[0]
        self.assertNotEqual(first, memory.associate_frame(1, [observation()])[0])
        with self.assertRaises(ValueError):
            memory.associate_frame(1, [])
        for kwargs in ({"dino_weight": -1}, {"threshold": float("nan")}, {"max_descriptors": 0}):
            with self.assertRaises(ValueError):
                AssociationConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
