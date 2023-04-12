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
from torch import nn
from torchmetrics import AUROC, AveragePrecision, F1Score, Metric, PrecisionRecallCurve

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
        layers: Union[int, list[int]],
        num_classes: int,
        T_max: int,
        dropout: Union[float, None] = None,
        lr: float = 0.0003,
        momentum: float = 0.01,
        wd: float = 0.01,
    ):
        """Initialize the Attention module following [1].

        Parameters
        ----------
        in_features : int
            Length of the ingoing feature vectors.
        layers : int, list[int]
            Length of the hidden feature vectors.
            If list of integers, make multiple hidden feature vectors and
            add dropout=0.5 in between by default.
        num_clases : int
            Number of classes the output should be.
        dropout_p : float, optional, default=0.5
            Probability of applying dropout to a parameter. Only applicable if
            `layers` is a list with length > 1.
        lr : float, default=0.0003
            Learning rate for optimizer.
        momentum : float, default=0.01
            Momentum for optimizer.
        T_max : int, default=1000
            Number of epochs. Used for the CosineAnnealingLR scheduler.
        wd : float, default=0.01
            Weight decay for optimizer.

        References
        [1] https://arxiv.org/abs/1802.04712
        """
        super(Attention, self).__init__()

        self.example_input_array = torch.Tensor(1, 1000, in_features)

        self.lr = lr
        self.momentum = momentum
        self.wd = wd
        self.T_max = T_max

        # DeepMIL specific initialization
        self.num_classes = num_classes
        self.L = in_features

        self.D: Union[int, list[int]]
        if isinstance(layers, list):
            if len(layers) > 1:
                self.D = layers
            else:
                self.D = layers[0]
        else:
            self.D = layers
        self.K = 1
        if isinstance(self.D, list):
            if dropout is None:
                dropout = 0.5
            attention_layers: list[nn.Module] = [
                nn.Linear(self.L, self.D[0]),
                nn.Tanh(),
            ]
            for i, curr_D in enumerate(self.D[1:]):
                prev_shape = self.D[i]
                attention_layers.extend(
                    [nn.Linear(prev_shape, curr_D), nn.Tanh(), nn.Dropout(dropout)]
                )
            attention_layers.append(
                nn.Linear(self.D[-1] if isinstance(self.D, list) else self.D, self.K)
            )
            self.attention = nn.Sequential(*attention_layers)
        else:
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
        self.pr_auc = AveragePrecision(task=task, num_classes=num_classes)  # type: ignore # noqa: 501

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _common_step(self, batch):
        """Perform a common step that is used in train/val/test."""
        x, y = batch["data"], batch["target"]
        y_hat, A = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y, A

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        loss, _, _, _ = self._common_step(batch)
        self.log("loss/train", loss, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        loss, y_hat, y, _ = self._common_step(batch)

        self.validation_output["target"].append(y)
        self.validation_output["prediction"].append(self.softmax(y_hat)[:, 1])
        self.validation_output["loss"].append(loss)

        self.log("loss/val", loss, batch_size=1)

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
        pr_auc = self.pr_auc(preds=prediction, target=target)  # type: ignore # although it should work according to docs. # noqa: 501

        # TODO Save these or do this afterwards from the patient-level outputs?
        precision, recall, thresholds = self.pr_curve(preds=prediction, target=target)

        # TODO Save the scores and cut-offs,
        # otherwise we can't do proper statistical testing.
        self.log(f"{prefix}_auc", auroc_score, logger=True, batch_size=1)
        self.log(f"{prefix}_f1", f1_score, logger=True, batch_size=1)
        self.log(f"{prefix}_pr_auc", pr_auc, logger=True, batch_size=1)

        if prefix == "test":
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

    def on_validation_epoch_end(self) -> None:
        """Procedure to run at the end of a validation epoch."""
        self.log_metrics(prefix="val", output=self.validation_output)
        self.validation_output = self._reset_output()

    def on_test_epoch_end(self) -> None:
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
                f"{self.trainer.log_dir}/output/{fold}/{case_id}_{img_id}_output.hdf5",
                "a",
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
        self, epoch: int, batch_idx: int, optimizer: torch.optim.Optimizer
    ) -> None:
        """Set gradients to None instead of zero.

        This improves performance.
        """
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.SGD(
            self.parameters(), self.lr, momentum=self.momentum, weight_decay=self.wd
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.T_max
        )
        return [optimizer], [scheduler]


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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
