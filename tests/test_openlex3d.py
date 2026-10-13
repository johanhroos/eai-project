import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from graph import shape
from utils.schema import Boxes, Graph, Masks, Run, encode_rle

HAS_OPENLEX3D = importlib.util.find_spec("openlex3d") is not None
if HAS_OPENLEX3D:
    import evaluate_openlex3d


class ObjectPointTests(unittest.TestCase):
    def test_thinning_keeps_one_point_per_cube(self):
        # One view of a 1 m square 2 m in front of the camera, in one mask of node 0
        size, depth = 200, 2.0
        K = np.array([[100.0, 0, size / 2], [0, 100.0, size / 2], [0, 0, 1]])
        square = np.zeros((size, size), bool)
        square[50:150, 50:150] = True
        masks = Masks(
            np.array([0], np.int32),
            np.zeros(1, np.int16),
            np.ones(1, np.int32),
            np.ones(1, np.float32),
            encode_rle(square[None]),
        )
        boxes = Boxes(np.zeros((1, 3), np.float32), np.ones((1, 3), np.float32), np.array([10_000], np.int32))
        get = lambda frame: (np.where(square, depth, 0.0).astype(np.float32), K, np.eye(4))  # noqa: E731
        run = Run("replica/room_0/00", 1, "sam3", "vggt")
        with mock.patch.object(shape, "view_source", return_value=get):
            dense = shape.object_points(run, masks, boxes, np.zeros(1, np.int32), max_points=None)[0]
            thin = shape.object_points(run, masks, boxes, np.zeros(1, np.int32), max_points=None, voxel=0.05)[0]
        cubes = np.floor(thin / 0.05).astype(int)
        self.assertEqual(len(np.unique(cubes, axis=0)), len(thin))  # one point per cube
        self.assertLess(len(thin), len(dense) / 4)
        np.testing.assert_allclose(thin.min(0), dense.min(0), atol=0.05)  # it still spans the square
        np.testing.assert_allclose(thin.max(0), dense.max(0), atol=0.05)


@unittest.skipUnless(HAS_OPENLEX3D, "needs the openlex3d extra")
class PredictionFormatTests(unittest.TestCase):
    def test_openlex3d_reads_each_point_with_its_object_embedding(self):
        from openlex3d.core.io import load_all_predictions, load_predicted_features

        rng = np.random.default_rng(0)
        points = np.concatenate([rng.uniform(0, 1, (500, 3)), rng.uniform(3, 4, (300, 3))]).astype(np.float32)
        index = np.r_[np.zeros(500), np.ones(300)].astype(np.int64)
        embeddings = rng.normal(size=(2, 8)).astype(np.float16)  # the siglip stage's dtype
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "method" / "replica" / "room0"
            evaluate_openlex3d.write_prediction(scene, points, index, embeddings)
            read_points, read_index, read_emb = load_all_predictions(Path(tmp) / "method" / "replica", "room0")
            cloud, features = load_predicted_features(str(scene), 0.05)
        np.testing.assert_allclose(read_points, points)
        np.testing.assert_array_equal(read_index, index)
        self.assertEqual(read_emb.dtype, np.float32)
        # Thinned to 5 cm, each point keeps the embedding of the object it came from
        far = cloud.point.positions.numpy()[:, 0] > 2
        np.testing.assert_allclose(features[far], np.broadcast_to(read_emb[1], (far.sum(), 8)))
        np.testing.assert_allclose(features[~far], np.broadcast_to(read_emb[0], ((~far).sum(), 8)))

    def test_a_node_is_its_box_with_its_points(self):
        rng = np.random.default_rng(0)
        points = np.concatenate([rng.uniform(0, 1, (1000, 3)), [[5.0, 5.0, 5.0]]]).astype(np.float32)  # and a streak
        graph = Graph(np.zeros(1, np.int32), np.zeros((1, 3), np.float32), np.full((1, 3), 0.5, np.float32))
        stored = evaluate_openlex3d.inside_boxes({0: points}, graph)[0]  # the graph stage's box
        self.assertTrue(((stored >= 0) & (stored <= 0.5)).all())
        self.assertGreater(len(stored), 50)
        drawn = evaluate_openlex3d.inside_boxes({0: points})[0]  # no stored box: drawn as shape.object_boxes does
        self.assertLess(drawn.max(), 1.0)  # the streak is outside
        self.assertGreater(len(drawn), 950)

    def test_replica_poses_need_no_alignment(self):
        fit, ate = evaluate_openlex3d.to_replica_frame(Run("replica/room_0/00", 20, "sam3", "gt"))
        np.testing.assert_array_equal(fit, np.eye(4))
        self.assertEqual(ate, 0.0)

    def test_method_names_tell_runs_apart(self):
        name = evaluate_openlex3d.method_name
        self.assertEqual(name(Run("replica/room_0/00", 20, "sam3", "vggt"), "graph"), "ours-sam3_vggt-stride20-graph")
        self.assertEqual(
            name(Run("replica/room_0/00", 10, "sam3", "gt", "indoor"), "memory"), "ours-sam3_indoor_gt-stride10-memory"
        )


if __name__ == "__main__":
    unittest.main()
