"""Invariants of the released MG-GBC implementation.

Each test states a property that the paper relies on, so that a change to the
code which breaks one of them fails here rather than in a table.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.arp_gb import ARPGB
from src.mg_gbc import MGGBC, rare_fragment_edges


def _two_blobs(seed: int, n: int = 300, d: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = n // 2
    return np.vstack(
        [
            rng.normal(0.0, 0.10, size=(half, d)),
            rng.normal(0.85, 0.10, size=(n - half, d)),
        ]
    )


class MGGBCInvariantTests(unittest.TestCase):
    def test_size_gate_is_fixed_across_keep_ablation(self) -> None:
        full = MGGBC(keep_rare=True)
        without_keep = MGGBC(keep_rare=False)
        self.assertEqual(full.size_gate, without_keep.size_gate)
        self.assertFalse(full.size_gate)

    def test_fragment_requires_closeness_relative_to_background(self) -> None:
        centers = np.array(
            [
                [0.0, 0.0],
                [0.9, 0.0],
                [1.0, 0.0],
            ]
        )
        radii = np.array([0.4, 0.4, 0.2])
        gamma = np.array([1.0, 0.9, 0.2])
        rare = np.array([True, True, False])
        edges = rare_fragment_edges(centers, radii, gamma, rare)
        self.assertFalse(edges[0, 1])

    def test_fragment_keeps_background_isolated_pair(self) -> None:
        centers = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [2.0, 0.0],
            ]
        )
        radii = np.array([0.05, 0.05, 0.2])
        gamma = np.array([1.0, 0.95, 0.2])
        rare = np.array([True, True, False])
        edges = rare_fragment_edges(centers, radii, gamma, rare)
        self.assertTrue(edges[0, 1])

    def test_no_rare_core_returns_the_baseline_partition(self) -> None:
        """Proposition 3: with no retained core the edit is empty."""
        X = _two_blobs(seed=0)
        reference = ARPGB(k=10, size_gate=False).fit_predict(X)
        corrected = MGGBC(k=10).fit_predict(X)
        self.assertEqual(MGGBC(k=10).fit_predict(X).tolist(), corrected.tolist())
        np.testing.assert_array_equal(corrected, reference)

    def test_repeated_runs_are_identical(self) -> None:
        X = _two_blobs(seed=1)
        first = MGGBC(k=10).fit_predict(X)
        second = MGGBC(k=10).fit_predict(X)
        np.testing.assert_array_equal(first, second)

    def test_labels_are_a_partition_of_every_point(self) -> None:
        X = _two_blobs(seed=2)
        model = MGGBC(k=10)
        labels = model.fit_predict(X)
        self.assertEqual(len(labels), len(X))
        self.assertEqual(model.n_clusters_, len(np.unique(labels)))
        self.assertGreaterEqual(int(labels.min()), 0)


if __name__ == "__main__":
    unittest.main()