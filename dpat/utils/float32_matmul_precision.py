"""Set float 32 matmul precision."""
from typing import Literal

import torch


def set_float32_matmul_precision(
    precision: Literal["medium", "high", "highest"]
) -> None:
    """Set float 32 matmul precision.

    Wrapper around torch.set_float32_matmul_precision.
    """
    torch.set_float32_matmul_precision(precision)
