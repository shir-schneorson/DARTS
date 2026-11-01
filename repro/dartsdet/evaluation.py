"""The ensemble statistic and the metrics every reported number is scored with.

Pixel vectors, shape (P,), are raveled in Fortran order (`scenes`). Labels are 1 (target),
0 (background) or `scenes.IGNORED`, scored neither way.
"""

import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve

from dartsdet.scenes import IGNORED

PF = 1e-3  # the operating false-alarm rate: PD at PF = 1e-3


def exceedance(statistic: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """-log of each pixel's exceedance probability (P,): the share of the reference pixels (P,)
    bool whose statistic is at least as high, at least one of them."""
    ranked = np.sort(statistic[reference])
    above = len(ranked) - np.searchsorted(ranked, statistic, side="left")
    return -np.log(np.maximum(above, 1) / len(ranked))


def ensemble(statistics: list[np.ndarray], reference: np.ndarray) -> np.ndarray:
    """The folds' statistics (P,) each, as the mean of their `exceedance` over the reference."""
    return np.mean([exceedance(s, reference) for s in statistics], 0)


def detection_metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """AUC, PD at PF and the standardized partial AUC up to PF of a score (P,) on labels (P,).

    PD is the largest PD at a threshold whose PF is at most `PF`; the partial AUC is McClish's,
    0.5 at chance and 1 perfect. The AUC leaves out the ROC's first segment, from (0, 0) to the
    highest threshold, as every baseline's own evaluation does.
    """
    scored = labels != IGNORED
    y, s = labels[scored], score[scored]
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    return {"auc": float(auc(fpr[1:], tpr[1:])), "pd": float(tpr[fpr <= PF].max()),
            "pauc": float(roc_auc_score(y, s, max_fpr=PF))}
