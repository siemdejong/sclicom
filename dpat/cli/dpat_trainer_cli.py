"""Provide dpat trainer cli."""
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from dpat.utils import enable_cudnn_auto_tuner, set_float32_matmul_precision


class DpatTrainerCLI(LightningCLI):
    """Trainer CLI to be used for dpat."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add custom arguments to the parser.

        ```yaml
        enable_cudnn_auto_tuner: false # Enable the cudnn auto tuner.
        set_float32_matmul_precision: highest # Set torch.set_float32_matmul_precision.
        ```
        """
        super().add_arguments_to_parser(parser)
        parser.add_argument("--enable_cudnn_auto_tuner", type=bool)
        parser.add_argument("--set_float32_matmul_precision", type=str)

    def before_instantiate_classes(self) -> None:
        """Routines to apply before trainer classes are instantiated."""
        super().before_instantiate_classes()
        enable_cudnn_auto_tuner(self.config["enable_cudnn_auto_tuner"])
        set_float32_matmul_precision(self.config["set_float32_matmul_precision"])
