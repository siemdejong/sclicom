"""Perform hyperparameter optimization."""

import warnings
from itertools import chain, product
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import optuna
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from ray import air, tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray_lightning import RayStrategy
from ray_lightning.tune import get_tune_resources

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


def train_tune(
    config: dict,
    datamodule: pl.LightningDataModule,
    num_epochs: int,
    num_gpus_per_trial: float,
    num_workers: int,
    num_cpus_per_worker: int,
    precision: _PRECISION_INPUT,
) -> None:
    """Train the model and report back to Ray.

    Parameters
    ----------
    config : dict
        (Hyper)parameter config with sampled parameters by Ray/Optuna.
    datamodule : pl.LightningDataModule
        Datamodule with data to feed the model.
    num_epochs : int
        Maximum number of epochs per trial.
    num_gpus_per_trial : float
        Number of gpus per trial. Can be fractional to share resources between trials.
    num_workers : int
        Number of workers that are running this function.
    num_cpus_per_worker : int
        Number of cpus available per worker.
    precision : str
        Precision for Lightning Fabric to work with.
    """
    model = VarAttention(
        # Fixed by combination of tile size and feature extractor.
        in_features=config["in_features"],
        hidden_features=config["hidden_features"],
        num_classes=config["num_classes"],
        dropout_p=config["dropout_p"],
        lr=config["lr"],
        T_max=num_epochs,
    )

    callback: pl.Callback = _TuneReportCallback(metrics="loss/val")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        enable_checkpointing=False,
        devices=np.ceil(num_gpus_per_trial)
        if num_gpus_per_trial
        else num_cpus_per_worker,
        accelerator="gpu" if num_gpus_per_trial else "cpu",
        callbacks=[callback],
        strategy=RayStrategy(
            num_workers=num_workers,
            num_cpus_per_worker=num_cpus_per_worker,
            use_gpu=bool(num_gpus_per_trial),
        ),
        precision=precision,
        enable_progress_bar=False,
    )

    trainer.logger.log_hyperparams(config)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["loss/val"].item()


if __name__ == "__main__":
    name = "tune_29-3-2023"
    num_trials = 10
    num_epochs = 10
    num_gpus_per_trial = 0.2
    num_workers = 2
    num_dataloader_workers = 8
    precision = "16-mixed"

    datamodule = PMCHHGH5DataModule(
        file_path=Path(
            "/scistor/guest/sjg203/projects/pmc-hhg/features/simclr-17-3-2023.hdf5"
        ),
        train_path="/scistor/guest/sjg203/projects/pmc-hhg/images-tif/splits/pilocytic-astrocytoma+medulloblastoma_pmchhg_train-subfold-0-fold-0.csv",  # noqa: E501
        val_path="/scistor/guest/sjg203/projects/pmc-hhg/images-tif/splits/pilocytic-astrocytoma+medulloblastoma_pmchhg_val-subfold-0-fold-0.csv",  # noqa: E501
        test_path="/scistor/guest/sjg203/projects/pmc-hhg/images-tif/splits/pilocytic-astrocytoma+medulloblastoma_pmchhg_test-subfold-0-fold-0.csv",  # noqa: E501
        num_workers=num_dataloader_workers,
        num_classes=2,
        balance=True,
    )

    _n_layers = np.arange(2)
    _powers = np.arange(9)
    config_space = {
        # Tunable
        "dropout_p": optuna.distributions.UniformDistribution(0.5, 0.95),
        "hidden_features": optuna.distributions.CategoricalDistribution(
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
        "lr": optuna.distributions.LogUniformDistribution(1e-5, 1e-2),
        # Constants
        "in_features": 1024,
        "num_classes": 2,
    }

    seed_configs_space = [
        config_space | {"dropout_p": 0.8, "hidden_features": [256, 8], "lr": 1e-4},
        config_space | {"dropout_p": 0.7, "hidden_features": [8], "lr": 1e-4},
    ]

    scheduler = ASHAScheduler(time_attr="training_iteration", max_t=num_epochs)
    search_alg = OptunaSearch(
        space=config_space,
        metric="val/loss",
        mode="min",
        points_to_evaluate=seed_configs_space,
    )

    tuner = tune.Tuner(
        tune.with_parameters(
            tune.with_resources(
                trainable=train_tune,
                resources=get_tune_resources(
                    num_workers=num_workers, use_gpu=bool(num_gpus_per_trial)
                ),
            ),
            datamodule=datamodule,
        ),
        tune_config=tune.TuneConfig(
            metric="loss/val",
            mode="min",
            num_samples=num_trials,
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=air.RunConfig(name=name),
    )

    results = tuner.fit()
