"""Train a feature extractor using self-supervised learning."""
import pathlib

import lightning.pytorch as pl

from dpat.cli import PreTrainCLI
from dpat.extract_features.trainer import PreTrainer


def main():
    """Train a feature extractor using self-supervised learning.

    Main entry point of feature extractor training.
    """
    # TODO: attach pl CLI to `dpat extract-features train`.
    cli = PreTrainCLI(
        model_class=pl.LightningModule,
        datamodule_class=pl.LightningDataModule,
        trainer_class=PreTrainer,
        subclass_mode_model=True,
        subclass_mode_data=True,
        run=False,
        parser_kwargs={
            "parser_mode": "omegaconf",
            "default_config_files": [
                str(
                    (
                        pathlib.Path(__file__).parent
                        / "../../dpat/configs/defaults/extract-features.yaml"
                    ).resolve()
                )
            ],
        },
    )

    cli.datamodule.prepare_data()
    cli.datamodule.setup("train")
    cli.trainer.fit(
        model=cli.model, train_dataloaders=cli.datamodule.train_dataloader()
    )
    # cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
