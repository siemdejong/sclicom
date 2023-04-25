"""Evaluation utilities."""

import numpy as np
import scipy.stats as st
from numpy.typing import NDArray


def compute_mean_and_confidence_interval(
    values: NDArray, confidence=0.95
) -> tuple[float, float]:
    """Compute the mean and confidence interval from a sequence of values."""
    n = len(values)
    m, se = np.mean(values), st.sem(values)
    h = se * st.t.ppf((1 + confidence) / 2, n - 1)
    return m, h
