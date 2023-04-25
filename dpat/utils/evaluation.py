"""Evaluation utilities."""

from typing import Sequence

import numpy as np
import scipy.stats as st


def compute_mean_and_confidence_interval(values: Sequence, confidence=0.95):
    """Compute the mean and confidence interval from a sequence of values."""
    values = np.array(values)
    n = len(values)
    m, se = np.mean(values), st.sem(values)
    h = se * st.t.ppf((1 + confidence) / 2, n - 1)
    return m, h
