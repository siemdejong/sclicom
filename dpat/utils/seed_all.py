"""Seed all RNGs."""

import random

import numpy as np
import torch


def seed_all(seed: int = 42) -> None:
    """Seed all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
