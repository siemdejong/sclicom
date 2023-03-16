"""Train a feature extractor using self-supervised learning."""
import pathlib

import lightning.pytorch as pl
import torch

from dpat.cli import MILTrainCLI
from dpat.mil.trainer import MILTrainer


def main():
    """Train a feature extractor using self-supervised learning.

    Main entry point of feature extractor training.
    """
    # TODO: attach pl CLI to `dpat mil train`.
    cli = MILTrainCLI(
        model_class=pl.LightningModule,
        datamodule_class=pl.LightningDataModule,
        trainer_class=MILTrainer,
        subclass_mode_model=True,
        subclass_mode_data=True,
        run=False,
        parser_kwargs={
            "parser_mode": "omegaconf",
            "default_config_files": [
                str(
                    (
                        pathlib.Path(__file__).parent
                        / "../../dpat/configs/defaults/varmil.yaml"
                    ).resolve()
                )
            ],
        },
    )

    compiled_model = torch.compile(cli.model)

    cli.trainer.fit(model=compiled_model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
