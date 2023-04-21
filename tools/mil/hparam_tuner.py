"""Perform hyperparameter optimization."""

import logging
import math
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Literal, Union

import lightning.pytorch as pl
import optuna
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from ray import air, tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

import dpat.mil.models
from dpat.data.pmchhg_h5_dataset import PMCHHGH5DataModule
from dpat.utils import seed_all

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore", "Choices for a categorical distribution should be a tuple of", UserWarning
)
os.environ[
    "TOKENIZERS_PARALLELISM"
] = "false"  # To disable tokenizer parallelism warnings.


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
    model_name: Literal["Attention", "VarAttention", "CCMIL"],
    datamodule: pl.LightningDataModule,
    num_classes: int,
    in_features: int,
    num_epochs: int,
    num_gpus_per_trial: float,
    num_cpus_per_trial: int,
    precision: _PRECISION_INPUT,
    seed: int = 42,
    model_kwargs: Union[dict[str, Any], None] = None,
) -> None:
    """Train the model and report back to Ray.

    Parameters
    ----------
    config : dict
        (Hyper)parameter config with sampled parameters by Ray/Optuna.
    model_name : str
        The name of the model to tune. Can be one of "Attention", "VarAttention', or
        "CCMIL".
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
    seed : int
        Seed for randomness.
    model_kwargs : dict[str, ...], default=None
        Will be passed to the model initializer.
        E.g. `llm_model` or `trainable_llm` for `CCMIL`.
    """
    seed_all(seed)

    layers = []
    for layer in range(config["_n_layers"]):
        _pow = config[f"_pow_nodes_layer_{layer}"]
        _pow_of_2 = 2**_pow
        layers.append(_pow_of_2)

    model = getattr(dpat.mil.models, model_name)(
        # Fixed by combination of tile size and feature extractor.
        in_features=in_features,
        layers=layers,
        num_classes=num_classes,
        dropout=config["dropout"],
        lr=config["lr"],
        momentum=config["momentum"],
        wd=config["wd"],
        T_max=num_epochs,
    )

    if model_kwargs is not None:
        model = partial(model, **model_kwargs)

    callback: pl.Callback = _TuneReportCallback(
        metrics=["loss/train", "loss/val", "val_f1", "val_pr_auc", "val_auc"]
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
        deterministic=True,
    )

    trainer.logger.log_hyperparams(config)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["loss/val"].item()


def define_by_run_func(
    trial: optuna.Trial,
    bounds_n_layers: tuple[int, int],
    bounds_pow: tuple[int, int],
    bounds_dropout: tuple[int, int],
    bounds_lr: tuple[int, int],
    bounds_momentum: tuple[int, int],
    bounds_wd: tuple[int, int],
) -> None:
    """Define-by-run function to create the search space."""
    _n_layers = trial.suggest_int("_n_layers", *bounds_n_layers)
    for _layer in range(_n_layers):
        trial.suggest_int(f"_pow_nodes_layer_{_layer}", *bounds_pow)

    trial.suggest_float("dropout", *bounds_dropout)

    trial.suggest_float("lr", *bounds_lr, log=True)
    trial.suggest_float("momentum", *bounds_momentum)
    trial.suggest_float("wd", *bounds_wd, log=True)


if __name__ == "__main__":
    restore_path = None  # "/home/sdejong/ray_results/tune_3-4-2023_1"

    name = "tune_13-4-2023_2"
    num_trials = 10
    min_epochs = 30
    num_epochs = 600
    num_gpus = 1
    num_gpus_per_trial = 0.1
    num_cpus_per_trial = 5
    num_dataloader_workers = num_cpus_per_trial
    precision = "bf16-mixed"

    seed = 42

    num_classes = 2
    in_features = 1024

    seed_all(seed)

    model_name = "CCMIL"

    tmpdir = Path(os.environ["TMPDIR"])
    fold = 0
    subfold = 0
    diagnosis = "medulloblastoma+pilocytic-astrocytoma"
    dataset = "pmchhg"
    h5 = "imagenet-21-4-2023.hdf5"

    splits_dirname = "splits-final"

    datamodule = PMCHHGH5DataModule(
        file_path=tmpdir / Path(f"features/{h5}"),
        train_path=tmpdir
        / Path(
            f"images-tif/{splits_dirname}/{diagnosis}_"
            f"{dataset}_train-subfold-{subfold}-fold-{fold}.csv"
        ),
        val_path=tmpdir
        / Path(
            f"images-tif/{splits_dirname}/{diagnosis}_"
            f"{dataset}_val-subfold-{subfold}-fold-{fold}.csv"
        ),
        num_workers=num_dataloader_workers,
        num_classes=num_classes,
        balance=True,
        clinical_context=True if model_name == "CCMIL" else False,
    )

    # Interesting discussion on wide+shallow vs narrow+deep:
    # https://stats.stackexchange.com/a/223637/384534
    # Deep and narrow can aid in generalizing.
    bounds_n_layers = (1, 4)
    bounds_pow = (1, 5)
    bounds_dropout = (0, 1)
    bounds_lr = (1e-5, 1e-2)
    bounds_momentum = (0, 1)
    bounds_wd = (1e-4, 1)
    define_by_run_space = partial(
        define_by_run_func,
        bounds_n_layers=bounds_n_layers,
        bounds_pow=bounds_pow,
        bounds_dropout=bounds_dropout,
        bounds_lr=bounds_lr,
        bounds_momentum=bounds_momentum,
        bounds_wd=bounds_wd,
    )

    seed_configs_space = [
        {"dropout": 0.7, "layers": [8], "lr": 1e-4},
        {"dropout": 0.75, "layers": [8, 4], "lr": 1e-4},
        {"dropout": 0.8, "layers": [8, 4, 2], "lr": 1e-4},
    ]

    scheduler = ASHAScheduler(
        time_attr="training_iteration", grace_period=min_epochs, max_t=num_epochs
    )
    search_alg = OptunaSearch(
        space=define_by_run_space,
        metric="loss/val",
        sampler=optuna.samplers.TPESampler(
            # Sample from a multivariate distribution,
            # not independent gaussian distributions.
            # Dropout and the layer structure are expected to be correlated tightly.
            multivariate=True,
            # To not run similar trials with distributed.
            # Penalties running configs around trials.
            constant_liar=True,
            # Because nodes per layer are depending on number of layers.
            group=True,
            # Because the search space is quite big
            n_startup_trials=num_trials / 5,
            seed=seed,
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
            model_name=model_name,
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
            max_concurrent_trials=int(num_gpus / num_gpus_per_trial),
        ),
        run_config=air.RunConfig(name=name),
    )

    if restore_path is not None:
        tuner = tune.Tuner.restore(
            path=restore_path,
            trainable=tune.with_parameters(
                tune.with_resources(
                    trainable=train_tune,
                    resources={"cpu": num_cpus_per_trial, "gpu": num_gpus_per_trial},
                ),
                model_name=model_name,
                datamodule=datamodule,
                num_classes=num_classes,
                in_features=in_features,
                num_epochs=num_epochs,
                num_gpus_per_trial=num_gpus_per_trial,
                num_cpus_per_trial=num_cpus_per_trial,
                precision=precision,
            ),
            resume_unfinished=False,
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
