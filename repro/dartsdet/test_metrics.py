"""The metrics, on hand-checked cases."""

import numpy as np

from dartsdet.data import IGNORED
from dartsdet.metrics import detection_metrics


def test_metrics_ignore_ignored_pixels() -> None:
    labels = np.array([0, 0, 0, 1, 1, IGNORED])
    score = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 5.0])
    assert detection_metrics(labels, score) == {"auc": 1.0, "pd": 1.0, "pauc": 1.0}


def test_pd_at_pf() -> None:
    # 2000 background pixels and 4 targets, two of them weak (5.5, 5.8). Two false alarms above
    # the weak targets are a PF of 1e-3, still allowed: PD 1. A third makes it 1.5e-3, and the
    # best allowed threshold then sits above the weak targets: PD 1/2.
    targets = [7.0, 8.0, 5.5, 5.8]
    labels = np.r_[np.zeros(2000), np.ones(4)]
    background = np.linspace(0, 1, 2000)
    background[:2] = [5.9, 6.0]
    assert detection_metrics(labels, np.r_[background, targets])["pd"] == 1.0
    background[2] = 6.5
    assert detection_metrics(labels, np.r_[background, targets])["pd"] == 0.5


def test_auc_leaves_out_the_first_roc_segment() -> None:
    # Two pixels tie at the top, one target and one background: the highest threshold already
    # sits at (PF, PD) = (1/2, 1/2). The full ROC's area is 0.625; without its first segment,
    # from (0, 0) to (1/2, 1/2), it is 0.5.
    labels = np.array([1, 0, 1, 0])
    score = np.array([1.0, 1.0, 0.5, 0.2])
    assert detection_metrics(labels, score)["auc"] == 0.5
