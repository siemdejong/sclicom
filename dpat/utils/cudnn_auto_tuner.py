"""Enable cudnn auto tuner."""
import torch


def enable_cudnn_auto_tuner(enable: bool = False):
    """Run short benchmark and select kernel with best performance.

    NOTE: Auto-tuner decisions may be non-deterministic.
    """
    torch.backends.cudnn.benchmark = enable
