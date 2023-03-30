"""Perform hyperparameter optimization."""

import logging
import math
import os
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

from dpat.data.pmchhg_h5_dataset import PMCHHGH5DataModule
from dpat.mil.models import VarAttention

logger = logging.getLogger(__name__)

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
    num_classes: int,
    in_features: int,
    num_epochs: int,
    num_gpus_per_trial: float,
    num_cpus_per_trial: int,
    precision: _PRECISION_INPUT,
) -> None:
    """Train the model and report back to Ray.

    Parameters
    ----------
    config : dict
        (Hyper)parameter config with sampled parameters by Ray/Optuna.
    datamodule : pl.LightningDataModule
        Datamodule with data to feed the model.
    num_classes : int
        Number of classes to train on.
    in_features : int
        Number of input features per tile.
    num_epochs : int
        Maximum number of epochs per trial.
    num_gpus_per_trial : float
        Number of gpus per trial. Can be fractional to share resources between trials.
    num_workers_per_trial : int
        Number of workers that are running this function.
    num_cpus_per_worker : int
        Number of cpus available per worker.
    precision : str
        Precision for Lightning Fabric to work with.
    """
    model = VarAttention(
        # Fixed by combination of tile size and feature extractor.
        in_features=in_features,
        hidden_features=config["hidden_features"],
        num_classes=num_classes,
        dropout_p=config["dropout_p"],
        lr=config["lr"],
        T_max=num_epochs,
    )

    callback: pl.Callback = _TuneReportCallback(
        metrics=["loss/val", "val_f1", "val_pr_auc", "val_auc"]

    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        enable_checkpointing=False,
        devices=math.ceil(num_gpus_per_trial)
        if num_gpus_per_trial
        else math.ceil(num_cpus_per_trial),
        accelerator="gpu" if num_gpus_per_trial else "cpu",
        callbacks=[callback],
        precision=precision,
        enable_progress_bar=False,
    )

    trainer.logger.log_hyperparams(config)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["loss/val"].item()


if __name__ == "__main__":
    name = "tune_29-3-2023_2"
    num_trials = 200
    min_epochs = 50
    num_epochs = 600
    num_gpus_per_trial = 0.1
    num_cpus_per_trial = 5
    num_dataloader_workers = num_cpus_per_trial
    precision = "bf16-mixed"

    num_classes = 2
    in_features = 1024

    tmpdir = Path(os.environ["TMPDIR"])
    fold = 0
    subfold = 0
    diagnosis = "medulloblastoma+pilocytic-astrocytoma"
    dataset = "pmchhg"
    h5 = "simclr-17-3-2023.hdf5"

    datamodule = PMCHHGH5DataModule(
        file_path=tmpdir / Path(f"features/{h5}"),
        train_path=tmpdir
        / Path(
            f"images-tif/splits/{diagnosis}_"
            f"{dataset}_train-subfold-{subfold}-fold-{fold}.csv"
        ),
        val_path=tmpdir
        / Path(
            f"images-tif/splits/{diagnosis}_"
            f"{dataset}_val-subfold-{subfold}-fold-{fold}.csv"
        ),
        num_workers=num_dataloader_workers,
        num_classes=num_classes,
        balance=True,
    )

    _n_layers = np.arange(2)
    _powers = np.arange(9)
    config_space = {
        "dropout_p": optuna.distributions.FloatDistribution(0.5, 0.95),
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
        "lr": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
    }

    seed_configs_space = [
        config_space | {"dropout_p": 0.8, "hidden_features": [256, 8], "lr": 1e-4},
        config_space | {"dropout_p": 0.7, "hidden_features": [8], "lr": 1e-4},
    ]

    scheduler = ASHAScheduler(
        time_attr="training_iteration", grace_period=min_epochs, max_t=num_epochs
    )
    search_alg = OptunaSearch(
        space=config_space,
        metric="loss/val",
        sampler=optuna.samplers.TPESampler(
            # Sample from a multivariate distribution,
            # not independent gaussian distributions.
            # Dropout and the layer structure are expected to be correlated tightly.
            multivariate=True,
            # To not run similar trials with distributed.
            # Penalties running configs around trials.
            constant_liar=True,
        ),
        mode="min",
        points_to_evaluate=seed_configs_space,
    )

    tuner = tune.Tuner(
        tune.with_parameters(
            tune.with_resources(
                trainable=train_tune,
                resources={"cpu": num_cpus_per_trial, "gpu": num_gpus_per_trial},
            ),
            datamodule=datamodule,
            num_classes=num_classes,
            in_features=in_features,
            num_epochs=num_epochs,
            num_gpus_per_trial=num_gpus_per_trial,
            num_cpus_per_trial=num_cpus_per_trial,
            precision=precision,
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

    logger.info(f"Best hyperparameters found were: {results.get_best_result().config}")
    logger.info(
        f"Best trial final loss: {results.get_best_result().metrics['loss/val']}"
    )
    logger.info(
        f"Best trial final f1 score: {results.get_best_result().metrics['val_f1']}"
    )
    logger.info(
        f"Best trial final PR-AUC: {results.get_best_result().metrics['val_pr_auc']}"
    )
