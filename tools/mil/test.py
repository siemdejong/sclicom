"""Train a feature extractor using self-supervised learning."""
import pathlib

import lightning.pytorch as pl

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

    cli.trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path="path")

    # folder=pathlib.Path("/scistor/guest/sjg203/projects/pmc-hhg/images-tif")

    # for batch in cli.datamodule.test_dataloader:
    #     pred, attention = cli.model(batch)
    # tile_x, tile_y = batch["tile_x"], batch["tile_y"]
    # case_id = batch["case_id"]
    # img_id = batch["img_id"]

    # img_path = np.array([img_path.glob(img_id) for img_path in folder.glob(case_id)]).flatten()[0]  # noqa: E501

    # image = Image.open(img_path)
    # for tile_xy in zip(tile_x, tile_y):

    # # Do min-max normalization, as in DeepMIL.
    # attention_normalized = (attention - torch.min(attention)) / (torch.max(attention) - torch.min(attention))  # noqa: E501


if __name__ == "__main__":
    main()
