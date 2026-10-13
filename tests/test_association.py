import unittest

import numpy as np

from graph.association import AssociationConfig, ObjectMemory, Observation
from graph.pipeline import replay
from utils.lift3d import Box3D
from utils.schema import Boxes, Features, Masks, TextAligned, encode_rle


def observation(descriptor=(1, 0), x=0, label="chair", sam_id=None, reliable=True):  # no SAM ID: DINO and location
    return Observation(
        np.array(descriptor, np.float32),
        Box3D(np.array([x, 0.0, 0.0]), np.array([x + 1, 1.0, 1.0]), 100),
        label,
        sam_id,
        np.array(descriptor, np.float32),
        reliable=reliable,
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


class TrackTests(unittest.TestCase):
    """A mask joins its SAM track's node unless the track jumped.

    Nodes merge when they match and were never seen in one frame.
    """

    def test_unreliable_view_follows_its_track_without_growing_the_box(self):
        memory = ObjectMemory()
        object_id = memory.associate_frame(0, [observation(sam_id=1)])[0]
        far = observation(descriptor=(0, 1), x=0.5, sam_id=1, reliable=False)  # would match nothing on its own
        self.assertEqual(memory.associate_frame(1, [far])[0], object_id)
        np.testing.assert_array_equal(memory.nodes[object_id].box.max, [1, 1, 1])
        self.assertEqual(len(memory.nodes[object_id].descriptors), 1)

    def test_track_is_cut_where_its_box_jumps(self):
        memory = ObjectMemory()
        first = memory.associate_frame(0, [observation(sam_id=1)])[0]
        jumped = memory.associate_frame(1, [observation(x=3, sam_id=1)])[0]  # 2 m from its last box, looking the same
        self.assertNotEqual(first, jumped)
        self.assertEqual(memory.tracks[1], jumped)
        self.assertEqual(memory.last_matches[0]["cut"]["gap"], 2)

    def test_track_is_cut_where_it_looks_different_and_moved_off_its_box(self):
        memory = ObjectMemory()
        first = memory.associate_frame(0, [observation(sam_id=1)])[0]
        moved = memory.associate_frame(1, [observation(descriptor=(0, 1), x=1.1, sam_id=1)])[0]  # 0.1 m off its box
        self.assertNotEqual(moved, first)
        self.assertAlmostEqual(memory.last_matches[0]["cut"]["gap"], 0.1, places=5)
        for config, cut in ((AssociationConfig(), False), (AssociationConfig(cut_dino_gap=None), True)):
            memory = ObjectMemory(config)
            first = memory.associate_frame(0, [observation(sam_id=1)])[0]
            # Looking different where its box still touches the last one: the same object from another side
            self.assertEqual(memory.associate_frame(1, [observation(descriptor=(0, 1), sam_id=1)])[0] != first, cut)

    def test_track_is_cut_where_another_node_matches_it_much_better(self):
        memory = ObjectMemory()
        mine, other = memory.associate_frame(0, [observation(sam_id=1), observation(x=1.2, sam_id=2)])
        # 0.2 m from its last box and looking the same, but exactly where the other node is
        self.assertEqual(memory.associate_frame(1, [observation(x=1.2, sam_id=1)])[0], other)
        self.assertIn("rival", memory.last_matches[0]["cut"])
        self.assertNotEqual(mine, other)

    def test_with_cut_after_2_a_one_frame_jump_stays_with_the_node(self):
        memory = ObjectMemory(AssociationConfig(cut_after=2))
        node = memory.associate_frame(0, [observation(sam_id=1)])[0]
        self.assertEqual(memory.associate_frame(1, [observation(x=3, sam_id=1)])[0], node)  # 2 m off: held, not cut
        self.assertEqual(memory.last_matches[0]["held"]["jumps"], 1)
        self.assertFalse(memory.last_matches[0]["reliable"])
        np.testing.assert_array_equal(memory.nodes[node].box.max, [1, 1, 1])  # the jumped mask adds no points
        self.assertEqual(memory.associate_frame(2, [observation(sam_id=1)])[0], node)  # back where it was
        self.assertIsNone(memory.last_matches[0]["held"])
        self.assertEqual(memory.associate_frame(3, [observation(x=3, sam_id=1)])[0], node)  # held again
        moved = memory.associate_frame(4, [observation(x=3, sam_id=1)])[0]  # the second jump in a row is cut
        self.assertNotEqual(moved, node)
        self.assertEqual(memory.tracks[1], moved)

    def test_frame_where_most_tracks_jump_cuts_nothing(self):
        memory = ObjectMemory()
        ids = memory.associate_frame(0, [observation(x=x, sam_id=k) for k, x in enumerate((0, 4, 8))])
        moved = memory.associate_frame(1, [observation(x=x + 0.5, sam_id=k) for k, x in enumerate((0, 4, 8))])
        np.testing.assert_array_equal(moved, ids)  # a bad camera pose, not three ID switches

    def test_new_track_merges_into_the_node_it_matches_once_it_is_reliable(self):
        memory = ObjectMemory()
        first = memory.associate_frame(0, [observation(sam_id=1)])[0]
        far = memory.associate_frame(1, [observation(descriptor=(0, 1), x=0.5, sam_id=2, reliable=False)])[0]
        self.assertNotEqual(first, far)  # too little evidence to match: a node of its own for now
        self.assertEqual(memory.associate_frame(2, [observation(sam_id=2)])[0], first)
        self.assertEqual(len(memory.nodes), 1)
        self.assertEqual(memory.find(far), first)
        self.assertEqual(memory.nodes[first].frames, {0, 1, 2})

    def test_nodes_seen_in_one_frame_never_merge(self):
        memory = ObjectMemory()
        ids = memory.associate_frame(0, [observation(sam_id=1), observation(x=0.2, sam_id=2)])
        self.assertNotEqual(*ids)
        np.testing.assert_array_equal(memory.associate_frame(1, [observation(sam_id=1), observation(sam_id=2)]), ids)
        self.assertEqual(len(memory.nodes), 2)

    def test_two_masks_of_one_frame_never_follow_their_tracks_to_one_node(self):
        memory = ObjectMemory()
        node = memory.associate_frame(0, [observation(sam_id=1)])[0]
        self.assertEqual(memory.associate_frame(1, [observation(sam_id=2)])[0], node)  # a new track matches it
        # Seen side by side, the two tracks are two objects: the one that matches the node less gets a node of its own
        ids = memory.associate_frame(2, [observation(x=0.2, sam_id=2), observation(sam_id=1)])
        self.assertEqual(ids[1], node)
        self.assertNotEqual(ids[0], node)
        self.assertEqual(memory.last_matches[0]["cut"]["taken"], node)
        self.assertEqual(memory.tracks, {1: node, 2: ids[0]})

    def test_mask_on_two_nodes_tracks_follows_the_older_without_merging_them(self):
        memory = ObjectMemory()
        first = memory.associate_frame(0, [observation(sam_id=1)])[0]
        second = memory.associate_frame(1, [observation(descriptor=(0, 1), x=5, sam_id=2)])[0]
        self.assertEqual(memory.associate_frame(2, [observation(sam_id=(1, 2))])[0], first)
        self.assertEqual(sorted(memory.nodes), [first, second])
        self.assertEqual(memory.tracks, {1: first, 2: first})

    def test_renumber_closes_the_gaps_merges_leave(self):
        memory = ObjectMemory()
        memory.associate_frame(0, [observation(sam_id=1), observation(x=4, sam_id=3)])
        memory.associate_frame(1, [observation(descriptor=(0, 1), sam_id=2, reliable=False)])  # node 2, merged later
        memory.associate_frame(2, [observation(sam_id=2)])
        self.assertEqual(sorted(memory.nodes), [0, 1])
        memory.associate_frame(3, [observation(x=8, sam_id=4)])  # node 3
        self.assertEqual(memory.renumber(), {0: 0, 1: 1, 2: 0, 3: 2})
        self.assertEqual(sorted(memory.nodes), [0, 1, 2])
        self.assertEqual(memory.tracks, {1: 0, 2: 0, 3: 1, 4: 2})


if __name__ == "__main__":
    unittest.main()
