"""Provide feature extractor trainer."""
import lightning.pytorch as pl


class PreTrainer(pl.Trainer):
    """Trainer for feature extractor."""

    def __init__(self, *args, **kwargs):
        """Initialize Pytorch Lightning Trainer."""
        super().__init__(*args, **kwargs)
