"""Provide multi instance learning trainer."""
import lightning.pytorch as pl


class MILTrainer(pl.Trainer):
    """Trainer for multi instance learning."""

    def __init__(self, *args, **kwargs):
        """Initialize Pytorch Lightning Trainer."""
        super().__init__(*args, **kwargs)
