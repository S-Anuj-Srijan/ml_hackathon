import unittest
import numpy as np

from task_c.decoder import RoadGraph, classifier_only, route_continuity, viterbi_decode, evaluate_ablation


class DecoderTests(unittest.TestCase):
    def setUp(self):
        # Directed chain: 0->1 (arc 0), 1->2 (arc 1), 2->3 (arc 2),
        # plus a disconnected arc 3->4 (arc 3).
        self.graph = RoadGraph(
            nodes=np.array([[0, 0], [1, 0], [2, 0], [3, 0], [9, 9]], dtype=float),
            arcs=np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=int),
        )

    def test_directed_connectivity(self):
        self.assertTrue(self.graph.is_directly_connected(0, 1))
        self.assertFalse(self.graph.is_directly_connected(1, 0))
        self.assertEqual(self.graph.arc_hops(0, 2, max_hops=3), 2)
        self.assertEqual(self.graph.arc_hops(0, 3, max_hops=3), 3)

    def test_viterbi_prefers_coherent_sequence(self):
        candidates = [
            [(0, 0.55), (3, 0.45)],
            [(3, 0.60), (1, 0.40)],
            [(2, 0.90), (3, 0.10)],
        ]
        result = viterbi_decode(
            self.graph, candidates, max_transition_hops=3,
            disconnected_penalty=20.0
        )
        self.assertEqual(result["predicted_arc_ids"], [0, 1, 2])

    def test_empty_candidate_and_gap(self):
        candidates = [[(0, 1.0)], [], [(2, 1.0)], [(3, 1.0)]]
        result = viterbi_decode(
            self.graph, candidates, timestamps=[0, 1, 30, 31],
            max_gap_seconds=5
        )
        self.assertEqual(result["predicted_arc_ids"], [0, None, 2, 3])
        self.assertEqual(result["empty_candidate_observations"], 1)

    def test_classifier_baseline_and_continuity(self):
        candidates = [[(0, .2), (3, .8)], [(1, .9)], [(2, .9)]]
        top1 = classifier_only(candidates)
        self.assertEqual(top1, [3, 1, 2])
        metrics = route_continuity(self.graph, [0, 1, 2])
        self.assertEqual(metrics["direct_continuity_rate"], 1.0)

    def test_ablation_comparison(self):
        candidates = [[(0, .2), (3, .8)], [(1, .9)], [(2, .9)]]
        result = evaluate_ablation(
            self.graph, candidates, [0, 1, 2],
            max_transition_hops=3, disconnected_penalty=20
        )
        self.assertAlmostEqual(result["classifier_only"]["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["classifier_plus_hmm"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
