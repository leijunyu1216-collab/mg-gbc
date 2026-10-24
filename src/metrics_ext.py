"""Extended cluster metrics that separate partition quality from the ACC bijection.

`cluster_acc` matches clusters to classes one-to-one with the Hungarian
algorithm.  A method that returns more clusters than there are classes can
therefore never score the surplus clusters, however pure they are: on the
UNSW-NB15 blocks the mass parked in unmatched clusters accounted for the whole
of MG-GBC's ACC deficit in 19 of 20 blocks, and for block 12 it matched the
deficit exactly (0.1536 against 0.1536).

`evaluate_extended` therefore reports, alongside the frozen metric bundle:

    many_to_one_ACC        each cluster votes for its majority class, so a finer
                           partition is not punished for being finer
    unmatched_cluster_share
                           fraction of points sitting in clusters the bijection
                           had to discard -- an upper bound on the ACC that
                           over-partitioning gives away
    ACC_plus_bijection_penalty
                           Hungarian ACC with that mass added back.  This is an
                           optimistic upper bound, not a corrected ACC: it
                           credits all unmatched-cluster mass as if correct.

src/metrics.py is deliberately left untouched: its SHA256 is frozen by the v1
temporal protocol, and editing it would make the v1 run unverifiable.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix

from .metrics import evaluate


def many_to_one_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy when every cluster is labelled by its majority class."""
    cm = contingency_matrix(np.asarray(y_true), np.asarray(y_pred))
    return float(cm.max(axis=0).sum() / len(y_true))


def bijection_penalty(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Mass and cluster count left unmatched by one-to-one Hungarian matching.

    The unmatched mass diagnoses sensitivity to surplus predicted clusters; it
    is not itself an error term that can be added back to recover true accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = contingency_matrix(y_true, y_pred)
    _, cols = linear_sum_assignment(-cm)
    unmatched_mass = int(cm.sum() - cm[:, cols].sum())
    return {
        "n_clusters_matched": float(len(cols)),
        "n_clusters_unmatched": float(cm.shape[1] - len(cols)),
        "unmatched_cluster_share": float(unmatched_mass / len(y_true)),
    }


def evaluate_extended(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = dict(evaluate(y_true, y_pred))
    out["many_to_one_ACC"] = many_to_one_acc(y_true, y_pred)
    out.update(bijection_penalty(y_true, y_pred))
    # Retained for compatibility with frozen UNSW outputs.  Interpret only as
    # an optimistic upper bound, never as a corrected or "true" ACC.
    out["ACC_plus_bijection_penalty"] = out["ACC"] + out["unmatched_cluster_share"]
    return out


EXTENDED_KEYS = [
    "many_to_one_ACC",
    "n_clusters_matched",
    "n_clusters_unmatched",
    "unmatched_cluster_share",
    "ACC_plus_bijection_penalty",
]
