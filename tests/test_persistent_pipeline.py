import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from graph.association import AssociationConfig, ObjectMemory, Observation
from graph.pipeline import load_memory, replay, save_result
from utils.lift3d import Box3D
from utils.schema import Boxes, Features, Graph, Masks, TextAligned, encode_rle, load_stage


def obs(x=0, dino=(1.0, 0.0), siglip=(1.0, 0.0)):
    return Observation(
        np.array(dino), Box3D(np.array([x, 0.0, 0.0]), np.array([x + 1, 1.0, 1.0]), 10), "chair", 99, np.array(siglip)
    )


def stages():
    masks = np.zeros((4, 12, 12), bool)
    masks[:, 2:8, 2:8] = True
    data = Masks(
        np.array([0, 0, 4, 4], np.int32),
        np.array([0, 1, 0, 1], np.int16),
        np.array([2, 3, 20, 21], np.int32),
        np.ones(4, np.float32),
        encode_rle(masks),
    )
    boxes = Boxes(np.zeros((4, 3), np.float32), np.ones((4, 3), np.float32), np.full(4, 100, np.int32))
    features = Features(np.tile(np.array([1, 0], np.float16), (4, 1)))
    siglip = TextAligned(features.dino.copy())
    return data, boxes, features, siglip, ["chair", "seat"]


class PipelineTests(unittest.TestCase):
    def test_malformed_modalities_and_dimensions_can_be_saved(self):
        from graph.pipeline import save_memory

        memory = ObjectMemory()
        memory.associate_frame(0, [obs()])
        malformed = Observation(np.array([1, 0, 0]), Box3D([4, 0, 0], [5, 1, 1], 10), siglip=np.array([1, 0, 0]))
        other = memory.associate_frame(1, [malformed])[0]
        self.assertFalse(memory.nodes[other].descriptors)
        self.assertIsNone(memory.nodes[other].siglip)
        with tempfile.TemporaryDirectory() as temporary:
            save_memory(Path(temporary), memory)
            self.assertEqual(len(load_memory(Path(temporary)).nodes), 2)
        invalid = Observation(["invalid"], Box3D(["invalid"] * 3, [1, 1, 1], 10))
        memory.associate_frame(2, [invalid])

    def test_siglip_takes_no_part_in_matching(self):
        memory = ObjectMemory()
        node_id = memory.associate_frame(0, [obs()])[0]
        self.assertEqual(memory.associate_frame(1, [obs(siglip=(0, 1))])[0], node_id)
        np.testing.assert_allclose(memory.nodes[node_id].siglip, np.array([1, 1]) / np.sqrt(2), rtol=1e-6)

    def test_normalization_switches_and_missing_cues(self):
        config = AssociationConfig(dino_weight=6, box_weight=2)
        np.testing.assert_allclose(config.effective_weights, [0.75, 0.25])
        memory = ObjectMemory(config)
        node_id = memory.associate_frame(0, [obs()])[0]
        missing = Observation(None, None, "chair", siglip=np.array([1.0, 0.0]))
        self.assertAlmostEqual(memory.score(missing, memory.nodes[node_id]), 0)
        disabled = AssociationConfig(enable_geometry=False)
        np.testing.assert_allclose(disabled.effective_weights, [1, 0])
        with self.assertRaises(ValueError):
            AssociationConfig(enable_dino=False, enable_geometry=False)
        negative = obs(dino=(-1, 0))
        self.assertAlmostEqual(memory.score(negative, memory.nodes[node_id]), 0.25)

    def test_candidate_gate_and_optional_center_distance(self):
        memory = ObjectMemory(AssociationConfig(threshold=0.1, max_candidate_distance=1))
        node_id = memory.associate_frame(0, [obs()])[0]
        self.assertEqual(memory.score(obs(x=3), memory.nodes[node_id]), -np.inf)
        self.assertNotEqual(memory.associate_frame(1, [obs(x=3)])[0], node_id)
        center_memory = ObjectMemory(
            AssociationConfig(
                dino_weight=0,
                box_weight=1,
                geometry_inside_weight=0,
                threshold=0.5,
                center_distance_scale=1,
            )
        )
        object_id = center_memory.associate_frame(0, [obs()])[0]
        shifted = obs(x=0.1)
        self.assertAlmostEqual(center_memory.score(shifted, center_memory.nodes[object_id]), np.exp(-0.1))
        self.assertEqual(center_memory.associate_frame(1, [shifted])[0], object_id)

    def test_replay_deduplicates_and_saves_resumable_memory(self):
        result = replay(*stages(), frames=[0, 1, 2, 3, 4])
        np.testing.assert_array_equal(result.graph.node, [0, 0, 0, 0])
        self.assertEqual([h["nodes"] for h in result.history], [1] * 5)
        self.assertEqual(result.memory.nodes[0].observation_count, 2)
        self.assertTrue(result.matches[-1]["matched"])
        self.assertEqual(result.matches[-1]["sam_ids"], [20, 21])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            save_result(path, result)
            np.testing.assert_array_equal(load_stage(path, Graph).node, result.graph.node)
            loaded = load_memory(path / "memory")
            self.assertEqual(loaded.config, result.memory.config)
            self.assertEqual(loaded.next_id, 1)
            self.assertAlmostEqual(
                loaded.score(obs(), loaded.nodes[0]), result.memory.score(obs(), result.memory.nodes[0])
            )
            loaded.associate_frame(5, [])
            self.assertEqual(loaded.associate_frame(6, [obs()])[0], 0)
            self.assertEqual(loaded.associate_frame(7, [obs(x=4)])[0], 1)
            self.assertEqual(json.loads((path / "history.json").read_text())[-1]["nodes"], 1)

    def test_parameter_change_and_prefix_replay(self):
        original = replay(*stages())
        stricter = replay(*stages(), config=AssociationConfig(threshold=1))
        self.assertEqual(len(original.memory.nodes), 1)
        self.assertEqual(len(stricter.memory.nodes), 2)
        prefix = replay(*stages(), until=0)
        np.testing.assert_array_equal(prefix.graph.node, [0, 0, -1, -1])
        self.assertEqual(prefix.memory.last_frame, 0)

    def test_edges_and_empty_memory_roundtrip(self):
        memory = ObjectMemory()
        memory.associate_frame(0, [obs(), obs(x=1.2), obs(x=4)])
        from graph.edges import build_edges
        from graph.visualize import memory_nodes

        near = [edge for edge in build_edges(memory_nodes(memory)) if edge.relation == "near"]
        self.assertEqual(len(near), 1)
        self.assertAlmostEqual(near[0].gap, 0.2, places=5)
        from graph.pipeline import save_memory

        with tempfile.TemporaryDirectory() as temporary:
            save_memory(Path(temporary), ObjectMemory())
            self.assertFalse(load_memory(Path(temporary)).nodes)

    def test_invalid_siglip_and_bounds(self):
        memory = ObjectMemory(AssociationConfig(threshold=0.39, max_descriptors=2))
        node_id = memory.associate_frame(0, [obs()])[0]
        invalid = obs(siglip=(np.nan, 0))
        memory.associate_frame(1, [invalid])
        node = memory.nodes[node_id]
        self.assertEqual(node.siglip_count, 1)
        for frame, descriptor in enumerate([(0, 1), (1, 1), (1, -1)], 2):
            memory.associate_frame(frame, [obs(dino=descriptor, siglip=descriptor)])
        self.assertLessEqual(len(node.descriptors), 2)


if __name__ == "__main__":
    unittest.main()
