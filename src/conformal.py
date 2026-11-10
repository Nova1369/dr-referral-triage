"""The class-wise conformal referral rule, in one place. C1, C2 and C3 all use these functions.

How it works, for a photo and a class c:

1. Score. s = 1 - p(c | photo), from the model's probabilities. A low score means the model finds c likely.
2. Threshold (classwise_thresholds). Take the calibration photos whose true class is c and their scores for c.
   With n of them, the threshold q_c is the ceil((n + 1)(1 - alpha))-th smallest score (conformal_quantile).
   A class needs at least 1/alpha - 1 photos (9 at alpha = 0.1); with fewer, q_c is infinite and c is always kept.
3. Prediction set. A new photo's set holds every class c with s_c <= q_c. When the calibration photos and the
   new photos are exchangeable, the set contains the true class with probability at least 1 - alpha,
   separately for each class (averaged over the draw of calibration photos).
4. Decision (decide_from_sets). Urgent if "urgent" is in the set; routine if the set is exactly {routine};
   no referral if it is exactly {no referral}; anything else goes to a human grader.

Why this meets the targets: a referable eye is sent home only when its set is exactly {no referral}, that is,
when its true class was left out. So 90 % coverage of the routine and urgent classes keeps the referable eyes
sent home at or below 10 %, and 90 % coverage of the urgent class means at least 90 % of urgent eyes are
referred urgently.

marginal_threshold is standard conformal prediction (baseline B4): one threshold for all classes, so its
guarantee is only an average over the classes.
"""
from __future__ import annotations

import math

import numpy as np

HOME, ROUTINE, URGENT, UNCERTAIN = 0, 1, 2, 3
DECISION_NAMES = {HOME: "no referral", ROUTINE: "routine", URGENT: "urgent", UNCERTAIN: "uncertain"}


def conformal_quantile(scores, alpha):
    """The ceil((n + 1)(1 - alpha))-th smallest score; infinite if there are too few scores."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def classwise_thresholds(P, y, alpha):
    """One threshold per class, each from the calibration photos of that class only.

    P: calibration probabilities (n photos x 3 classes); y: their true classes (0, 1, 2)."""
    S = 1.0 - P
    return np.array([conformal_quantile(S[y == c, c], alpha) for c in range(P.shape[1])])


def marginal_threshold(P, y, alpha):
    """A single threshold from every calibration photo's score for its own class (baseline B4)."""
    S = 1.0 - P
    return conformal_quantile(S[np.arange(len(y)), y], alpha)


def decide_from_sets(in_set):
    """Referral decision from the prediction sets (in_set: n photos x 3 booleans, one column per class)."""
    urgent = in_set[:, 2]
    only_routine = in_set[:, 1] & ~in_set[:, 0] & ~urgent
    only_none = in_set[:, 0] & ~in_set[:, 1] & ~urgent
    d = np.full(len(in_set), UNCERTAIN)
    d[only_none] = HOME
    d[only_routine] = ROUTINE
    d[urgent] = URGENT
    return d
