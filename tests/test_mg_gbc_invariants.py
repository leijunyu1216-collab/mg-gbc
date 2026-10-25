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


def _random_balls(seed: int, m: int, d: int, n_rare: int):
    """A random ball configuration with a fixed rare/background split."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(m, d))
    radii = rng.uniform(0.02, 0.30, size=m)
    gamma = rng.uniform(0.05, 1.0, size=m)
    rare = np.zeros(m, dtype=bool)
    rare[rng.choice(m, size=n_rare, replace=False)] = True
    return centers, radii, gamma, rare


class CorridorVacuityTests(unittest.TestCase):
    """The corridor clause at the registered constants.

    Lemma (corridor vacuity) in Section 3.3: with clearance factor
    ``beta_o`` and closeness factor ``beta_c`` satisfying
    ``beta_o <= 1/beta_c - 1/2``, any pair that passes the closeness test
    has its midpoint at least ``(1/beta_c - 1/2) * d`` from every
    background centre, so the corridor test cannot reject it.  At the
    registered ``beta_c = beta_o = 0.5`` the required clearance is ``0.5 d``
    and the available clearance is ``1.5 d``.  These tests check the two
    consequences the paper reports: the witness count is irrelevant, and
    the clause itself is redundant.
    """

    def test_witness_count_does_not_change_the_edit(self) -> None:
        for seed in range(12):
            centers, radii, gamma, rare = _random_balls(seed, m=14, d=3, n_rare=4)
            reference = rare_fragment_edges(centers, radii, gamma, rare)
            for w in (1, 2, 3, 5, 13, None):
                with self.subTest(seed=seed, witnesses=w):
                    got = rare_fragment_edges(centers, radii, gamma, rare,
                                              corridor_witnesses=w)
                    np.testing.assert_array_equal(got, reference)

    def test_corridor_clause_is_redundant_at_the_registered_constants(self) -> None:
        """A corridor factor that can never reject gives the same edge set."""
        for seed in range(12):
            centers, radii, gamma, rare = _random_balls(seed, m=14, d=3, n_rare=4)
            reference = rare_fragment_edges(centers, radii, gamma, rare)
            with self.subTest(seed=seed):
                vacuous = rare_fragment_edges(centers, radii, gamma, rare,
                                             corridor_clearance_factor=-1.0)
                np.testing.assert_array_equal(vacuous, reference)

    def test_the_clause_becomes_operative_above_closeness_one(self) -> None:
        """The lemma's hypothesis is sharp: at beta_c = 2 the clause rejects.

        Two rare centres at distance 1 with a background centre 0.4 above
        their midpoint.  Each rare centre is 0.640 from the background one,
        so the pair is ``closer`` at beta_c = 2 (1 < 1.28) and is not
        ``closer`` at the registered beta_c = 0.5 (1 > 0.32).  The midpoint
        is 0.4 from the background centre, below the 0.5 clearance the
        clause demands, so at beta_c = 2 the clause is what rejects the
        pair, and disabling it adds the edge.
        """
        centers = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.4]])
        radii = np.array([0.05, 0.05, 0.30])
        gamma = np.array([1.00, 0.95, 0.20])
        rare = np.array([True, True, False])

        at_registered = rare_fragment_edges(centers, radii, gamma, rare)
        self.assertFalse(at_registered[0, 1])

        loose = rare_fragment_edges(centers, radii, gamma, rare,
                                    background_closeness_factor=2.0)
        self.assertFalse(loose[0, 1])

        loose_vacuous = rare_fragment_edges(centers, radii, gamma, rare,
                                            background_closeness_factor=2.0,
                                            corridor_clearance_factor=-1.0)
        self.assertTrue(loose_vacuous[0, 1])


if __name__ == "__main__":
    unittest.main()