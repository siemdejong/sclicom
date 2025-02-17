"""Provide feature extractor trainer."""
import lightning.pytorch as pl


class PreTrainer(pl.Trainer):
    """Trainer for feature extractor."""

    def __init__(self, *args, **kwargs):
        """Initialize Pytorch Lightning Trainer."""
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        """Add arguments to parser and link configurations."""
        parser.link_arguments(
            "trainer.max_epochs", "model.init_args.scheduler.init_args.T_max"
        )
