"""Geometry/alignment and artifact invariants for the Part A handoff."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from data_pipeline import Network, candidate_features, align_route, project, FEATURE_COLUMNS


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.net = Network(np.array([[8., 50.], [8.01, 50.], [8.02, 50.]]),
                           np.array([[0, 1], [1, 0], [1, 2], [2, 1]]))

    def test_long_arc_midpoint_and_direction(self):
        track = np.array([[8.004, 50., 1], [8.005, 50., 2]])
        f, obs = candidate_features(self.net, track, k=4, radius=20)
        self.assertEqual(set(f.arc_id), {0, 1})
        self.assertTrue((obs.candidate_count == 2).all())
        forward = f[(f.obs_id == 1) & (f.arc_id == 0)].iloc[0]
        reverse = f[(f.obs_id == 1) & (f.arc_id == 1)].iloc[0]
        self.assertLess(forward.heading_difference_rad, .01)
        self.assertGreater(reverse.heading_difference_rad, 3.13)
        self.assertAlmostEqual(forward.offset_fraction, .5, places=3)

    def test_no_candidate_preserves_observation(self):
        f, obs = candidate_features(self.net, np.array([[9., 51., 1]]), radius=1)
        self.assertTrue(f.empty)
        self.assertEqual(obs.candidate_count.tolist(), [0])

    def test_reference_alignment_follows_route_occurrences(self):
        track = np.array([[8.002, 50., 1], [8.009, 50., 2], [8.015, 50., 3]])
        labels, occurrence, residual = align_route(self.net, project(track[:, :2], self.net.transformer),
                                                   track[:, 2], np.array([0, 2]))
        self.assertEqual(labels.tolist(), [0, 0, 2])
        self.assertTrue((np.diff(occurrence) >= 0).all())
        self.assertLess(residual.max(), 1)

    @unittest.skipUnless(Path('artifacts/data/features.parquet').exists(), 'Run pipeline first')
    def test_full_artifact_contract(self):
        f = pd.read_parquet('artifacts/data/features.parquet').rename(columns={'split': 'partition', 'obs_idx': 'obs_id'})
        f['partition'] = f['partition'].replace({'student': 'student_test'})
        o = pd.read_parquet('artifacts/data/observations.parquet')
        keys = ['partition', 'trip_id', 'obs_id']
        self.assertFalse(f.duplicated(keys + ['arc_id']).any())
        self.assertFalse(o.duplicated(keys).any())
        self.assertEqual(o.groupby('partition').trip_id.nunique().to_dict(), {'student_test': 10, 'train': 80})
        self.assertTrue((f.groupby(keys).label.sum() <= 1).all())
        counts = f.groupby(keys).size().reindex(pd.MultiIndex.from_frame(o[keys]), fill_value=0)
        np.testing.assert_array_equal(counts.to_numpy(), o.candidate_count)
        hits = f.groupby(keys).label.max().reindex(pd.MultiIndex.from_frame(o[keys]), fill_value=0)
        np.testing.assert_array_equal(hits.to_numpy().astype(bool), o.candidate_hit)
        self.assertTrue(np.isfinite(f[[c for c in FEATURE_COLUMNS if c not in ['speed_mps', 'heading_difference_rad']]]).all().all())
        self.assertTrue((f.distance_m <= 300 + 1e-4).all())
        self.assertTrue((f.offset_fraction.between(0, 1)).all())


if __name__ == '__main__':
    unittest.main()
