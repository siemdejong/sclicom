"""Provide VarMIL [1] adapted from [2].

References
----------
[1] https://github.com/NKI-AI/dlup-lightning-mil
[2] https://doi.org/10.1016/j.media.2022.102464
"""
import json
import warnings
from pathlib import Path
from typing import Literal, Union

import h5py
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import nn
from torchmetrics import AUROC, F1Score, Metric, PrecisionRecallCurve

# Of course the targets are equal per batch all the time, because of MIL.
# Suppress this warning.
warnings.filterwarnings(
    action="ignore",
    message="No negative samples in targets",
    category=UserWarning,
    module="torchmetrics",
)


class Attention(pl.LightningModule):
    """Attention pooling for MIL [1].

    Adapted from [2]

    References
    ----------
    [1] https://arxiv.org/abs/1802.04712
    [2] https://github.com/NKI-AI/dlup-lightning-mil
    """

    def _reset_output(self):
        """Reset the batch output."""
        output = {"loss": [], "target": [], "prediction": []}
        return output

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.CosineAnnealingLR,  # type: ignore # noqa: E501
    ):
        """Initialize the Attention module following [1].

        Parameters
        ----------
        in_features : int
            Length of the ingoing feature vectors.
        hidden_features : int
            Length of the hidden feature vectors.
        num_clases : int
            Number of classes the output should be.
        optimizer : OptimizerCallable, default=Adam
            Optimizer to use.
        scheduler : LRSchedulerCallable, default=CosineAnnealingLR
            Scheduler to use.

        References
        [1] https://arxiv.org/abs/1802.04712
        """
        super(Attention, self).__init__()

        self.example_input_array = torch.Tensor(1, 1000, 512)

        self.optimizer = optimizer
        self.scheduler = scheduler

        # DeepMIL specific initialization
        self.num_classes = num_classes
        self.L = in_features
        self.D = hidden_features
        self.K = 1
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, self.num_classes))

        self.loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        # Initialize validation output
        self.validation_output = self._reset_output()
        self.test_output = self._reset_output()

        # Initialize metrics
        task: Literal["binary", "multiclass", "multilabel"] = (
            "binary" if num_classes == 2 else "multiclass"
        )
        self.auroc: Metric = AUROC(task=task, num_classes=num_classes)  # type: ignore
        self.f1: Metric = F1Score(task=task, num_classes=num_classes)  # type: ignore
        self.pr_curve: Metric = PrecisionRecallCurve(  # type: ignore
            task=task, num_classes=num_classes
        )

    def forward(self, x):
        """Calculate prediction and attention vector."""
        # Since we have batch_size = 1,
        # squeezes from (1,num,features) to (num, features).
        H = x.squeeze(0)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_hat = self.classifier(M)

        return Y_hat, A

    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        return [optimizer], [scheduler]

    def _common_step(self, batch):
        """Perform a common step that is used in train/val/test."""
        x, y = batch["data"], batch["target"]
        y_hat, A = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y, A

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        loss, _, _, _ = self._common_step(batch)
        self.log("loss/train", loss, batch_size=1, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        loss, y_hat, y, _ = self._common_step(batch)

        self.validation_output["target"].append(y)
        self.validation_output["prediction"].append(self.softmax(y_hat)[:, 1])
        self.validation_output["loss"].append(loss)

        self.log("loss/val", loss, batch_size=1, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        loss, y_hat, y, A = self._common_step(batch)

        self.test_output["target"].append(y)
        self.test_output["prediction"].append(self.softmax(y_hat)[:, 1])
        self.test_output["loss"].append(loss)

        self.save_output(batch, A, y_hat, fold="test")
        return loss

    def log_metrics(self, prefix: str, output: dict[str, list[torch.Tensor]]):
        """Log custom metrics.

        Parameters
        ----------
        prefix : str
            Prefix of the log names. E.g. train/val/test.
        output : dict[str, list[torch.Tensor]]
            Outputs previously saved to log. The function logs
            "target" and "prediction" keys.
        """
        target = torch.ShortTensor(output["target"])
        prediction = torch.Tensor(output["prediction"])

        auroc_score = self.auroc(preds=prediction, target=target)
        f1_score = self.f1(preds=prediction, target=target)

        # TODO Save these or do this afterwards from the patient-level outputs?
        precision, recall, thresholds = self.pr_curve(preds=prediction, target=target)

        # TODO Save the scores and cut-offs,
        # otherwise we can't do proper statistical testing.
        self.log(
            f"{prefix}_auc",
            auroc_score,
            prog_bar=True,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_f1",
            f1_score,
            prog_bar=True,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

        if False:
            # if self.trainer.save_validation_output_to_disk:
            if not (Path(self.trainer.log_dir) / f"output/{prefix}").is_dir():
                Path.mkdir(
                    Path(self.trainer.log_dir) / f"output/{prefix}", parents=True
                )

            metrics_to_save = {
                "auc": float(auroc_score),
                "f1": float(f1_score),
                "prcurve": {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": thresholds.tolist(),
                },
            }

            with open(
                Path(self.trainer.log_dir) / f"output/{prefix}/metrics.json", "w"
            ) as f:
                f.write(json.dumps(metrics_to_save))

    def validation_epoch_end(self, validation_step_outputs) -> None:
        """Procedure to run at the end of a validation epoch."""
        self.log_metrics(prefix="val", output=self.validation_output)
        self.validation_output = self._reset_output()

    def test_epoch_end(self, test_step_outputs) -> None:
        """Procedure to run at the end of a test epoch."""
        self.log_metrics(prefix="test", output=self.test_output)
        self.test_output = self._reset_output()

    def save_output(
        self,
        batch: dict[str, torch.Tensor],
        As: torch.Tensor,
        y_hats: torch.Tensor,
        fold: str,
    ) -> None:
        """Save output to disk.

        Writes attention vector and prediction along with metadata to an HDF5 file.

        Parameters
        ----------
        batch : torch.Tensor
            Batch of size 1 with feature vectors and metadata.
        As : torch.Tensor
            Computed attention vector.
        y_hats : torch.Tensor
            Computed prediction.
        fold : str
            Fold (train/val/test) of the output to be saved.
        """
        if self.trainer.log_dir is None:
            log_dir: Path = Path.cwd()
        else:
            log_dir = Path(self.trainer.log_dir)

        if not (log_dir / f"output/{fold}").is_dir():
            (log_dir / f"output/{fold}").mkdir(parents=True)

        batch_size = len(batch["target"])  # could be any other

        for i in range(batch_size):
            target, case_id, img_id = (
                batch["target"][i],
                batch["case_id"][i],
                batch["img_id"][i],
            )

            # This data is shared among BC and CRC dataset
            hf = h5py.File(
                f"{self.trainer.log_dir}/output/{fold}/{img_id}_output.hdf5", "a"
            )
            hf["img_id"] = img_id
            hf["case_id"] = case_id
            hf["attention"] = As[i].cpu()
            hf["target"] = target.cpu()
            hf["prediction"] = (
                torch.nn.functional.softmax(y_hats.cpu(), dim=1)[:, 1].cpu().tolist()
            )

            hf["tile_x"] = batch["tile_x"][0].cpu()
            hf["tile_y"] = batch["tile_y"][0].cpu()
            hf["tile_mpp"] = batch["tile_mpp"][0].cpu()
            hf["tile_region_index"] = batch["tile_region_index"][0].cpu()

            hf.close()

    def optimizer_zero_grad(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int,
    ) -> None:
        """Set gradients to None instead of zero.

        This improves performance.
        """
        optimizer.zero_grad(set_to_none=True)


class VarAttention(Attention):
    """Compute the variance attention introduced in of DeepSMILE [1].

    Adapted from [2].

    References
    ----------
    [1] https://doi.org/10.1016/j.media.2022.102464
    [2] https://github.com/NKI-AI/dlup-lightning-mil
    """

    def __init__(self, *args, **kwargs):
        """Overwrite classifier attribute of `Attention`.

        `Attention` only does mean pooling, while `VarAttention` also does
        variance pooling, which is concatenated to the mean vector.

        Parameters
        ----------
        See `Attention`.
        """
        super(VarAttention, self).__init__(*args, **kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(
                2 * self.L * self.K, self.num_classes
            )  # 2x since we also have variance
        )

    def compute_weighted_var(
        self, A: torch.Tensor, H: torch.Tensor, M: Union[torch.Tensor, None] = None
    ):
        """Compute the weighted variance following [1].

        Parameters
        ----------
        A : `torch.Tensor`
            Tensor with attention factors for feature vectors in `H`.
        H : `torch.Tensor`
            Tensor with feature vectors.
        M : `torch.Tensor`, default=None
            Weighted average computed by `A`x`H`. If not given, it will be computed.

        References
        ----------
        [1] https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
        """
        if M is None:
            M = torch.mm(A, H)

        # TODO Now implemented to work with output as given above which is only for
        # batch size of 1.

        A, H = A.unsqueeze(2), H.unsqueeze(0)
        # A: Attention (weight):    batch x instances x 1
        # H: Hidden:                batch x instances x channels
        H = H.permute(0, 2, 1)  # batch x channels x instances

        # M: Weighted average:      batch x channels
        M = M.unsqueeze(dim=2)  # batch x channels x 1
        # ---> S: weighted stdev:   batch x channels

        # N is non-zero weights for each bag: batch x 1

        N = (A != 0).sum(dim=1)

        upper = torch.einsum("abc, adb -> ad", A, (H - M) ** 2)  # batch x channels
        lower = ((N - 1) * torch.sum(A, dim=1)) / N  # batch x 1

        # Square root leads to infinite gradients when input is 0
        # Solution: No square root, or add eps=1e-8 to the input
        # But adding the eps will still lead to a large gradient from the sqrt through
        # these 0-values.
        # Whether we look at stdev or variance shouldn't matter much,
        # so we choose to go for the variance.
        S = upper / lower

        return S

    def forward(self, x: torch.Tensor):
        """Calculate prediction and attention.

        Parameters
        ----------
        x : `torch.Tensor`
            Batch of size 1 with N feature vectors.
        """
        # Since we have batch_size = 1,
        # squeezes from (1,num,features) to (num, features)
        H = x.squeeze(0)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        S = self.compute_weighted_var(A, H)

        # Concatenate the two tensors among the feature dimension,
        # giving a twice as big feature.
        MS = torch.cat((M, S), dim=1)

        Y_hat = self.classifier(MS)
        return Y_hat, A
