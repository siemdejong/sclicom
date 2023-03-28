"""Perform hyperparameter optimization."""

import warnings
from itertools import chain, product
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from dpat.data.pmchhg_h5_dataset import PMCHHGH5DataModule
from dpat.mil.models import VarAttention

warnings.filterwarnings(
    "ignore", "Choices for a categorical distribution should be a tuple of", UserWarning
)


class _TuneReportCallback(TuneReportCallback, pl.Callback):
    """Workaround for issue [1].

    Ray is still using legacy pytorch_lightning imports. [2] provides a workaround.

    References
    ----------
    [1] https://github.com/ray-project/ray/issues/33426
    [2] https://github.com/ray-project/ray/issues/33426#issuecomment-1481951889
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def train(config: dict, datamodule) -> None:
    """Train the model and report back to Ray.

    Parameters
    ----------
    config : dict
        (Hyper)parameter config with sampled parameters by Ray/Optuna.
    datamodule : pl.DataModule
        Datamodule with data to feed the model.
    in_features : int
        Number of input features to the model. Must match the number of features in
        the datamodule.
    num_classes : int
        Number of classes that the model can predict.
    """
    model = VarAttention(
        in_features=config[
            "in_features"
        ],  # Fixed by combination of tile size and feature extractor.
        hidden_features=config["hidden_features"],
        num_classes=config["num_classes"],
        dropout_p=config["dropout_p"],
        lr=config["lr"],
        T_max=config["T_max"],
    )
    callback: pl.Callback = _TuneReportCallback(metrics="loss/val")

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        devices=1,
        accelerator="gpu",
        callbacks=[callback],
        strategy="ddp",
        precision=config["precision"],
    )

    trainer.logger.log_hyperparams(config)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["loss/val"].item()


if __name__ == "__main__":
    num_samples = 100

    datamodule = PMCHHGH5DataModule(
        file_path=Path(
            "/scistor/guest/sjg203/projects/pmc-hhg/features/simclr-17-3-2023.hdf5"
        ),
        train_path="/scistor/guest/sjg203/projects/pmc-hhg/images-tif/splits/pilocytic-astrocytoma+medulloblastoma_pmchhg_train-subfold-0-fold-0.csv",  # noqa: E501
        val_path="/scistor/guest/sjg203/projects/pmc-hhg/images-tif/splits/pilocytic-astrocytoma+medulloblastoma_pmchhg_val-subfold-0-fold-0.csv",  # noqa: E501
        test_path="/scistor/guest/sjg203/projects/pmc-hhg/images-tif/splits/pilocytic-astrocytoma+medulloblastoma_pmchhg_test-subfold-0-fold-0.csv",  # noqa: E501
        num_workers=0,
        num_classes=2,
        balance=True,
    )

    _n_layers = np.arange(3)
    _powers = np.arange(10)
    config = {
        # Tunable
        "dropout_p": tune.uniform(0.5, 0.95),
        "hidden_features": tune.choice(
            # This right here, is a fine art of spaghetti
            # to get a semi-flat array, with sorted powers of 2,
            # with length depending on number of layers.
            [
                sorted(np.unique(item), reverse=True)
                for item in chain(
                    *[
                        [
                            sorted([2**a for a in prod], reverse=True)
                            for prod in product(_powers + 1, repeat=n)
                        ]
                        for n in _n_layers + 1
                    ]
                )
            ]
        ),
        "lr": tune.loguniform(1e-5, 1e-2),
        # Constants
        "in_features": 1024,
        "num_classes": 2,
        "T_max": 500,
        "precision": "16-mixed",
    }

    scheduler = ASHAScheduler(time_attr="training_iteration", max_t=config["T_max"])
    search_alg = OptunaSearch()

    tuner = tune.Tuner(
        tune.with_parameters(train, datamodule=datamodule),
        tune_config=tune.TuneConfig(
            metric="loss/val",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        param_space=config,
    )

    results = tuner.fit()
