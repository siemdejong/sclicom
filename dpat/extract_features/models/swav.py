"""Provide SwAV model."""
from typing import Union

import lightning.pytorch as pl
import torch
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch import nn

from dpat.types import VisionBackbone


class SwAV(pl.LightningModule):
    """Swapping Augmented Views SSL.

    Depends on Lightly.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 64,
        n_prototypes: int = 16,
        backbone: str = "shufflenetv2_w1",
        pretrained: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.CosineAnnealingLR,  # type: ignore # noqa: E501
    ):
        """Build SwAV model.

        Parameters
        ----------
        input_dim : int, default=512
            Input dimension of the SwAV projection head.
        hidden_dim : int, default=256
            Dimension of the hidden layer in the SwAV projection head.
        output_dim : int, default=64
            Output dimension of the SwAV projection head. Will also be used as input
            dimension of SwAV prototypes.
        n_prototypes : int, default=16
            Number of prototypes to let SwAV work with.
        backbone : str, default=shufflenetv2_w1
            Backbone to pretrain. Can be any model provided by pytorchcv,
            e.g. `resnet18_wd4|shufflenetv2_w1`.
        pretrained : bool, default=False,
            Specify if the model should already be trained on ImageNet.
        optimizer : `OptimizerCallable`, default=torch.optim.Adam
            Optimizer to use.
        optimizer_kwargs : dict[str, Any] | None, default = None
            Kwargs to configure optimizer with. E.g. `lr`, `betas`, etc.
        scheduler : `SchedulerCallable`, default=CosineAnnealingLR
            Scheduler to use.
        scheduler_kwargs : dict[str, Any] | None, default=None
            Kwargs to configure scheduler with. E.g. `T_max`.
        """
        super().__init__()

        self.example_input_array = torch.Tensor(32, 3, 224, 224)

        self.optimizer = optimizer
        self.scheduler = scheduler

        # Get any vision model from pytorchcv as backbone.
        vision_backbone: VisionBackbone = ptcv_get_model(
            backbone, pretrained=pretrained
        )

        self.backbone = nn.Sequential(*list(vision_backbone.children())[:-1])
        self.projection_head = SwaVProjectionHead(input_dim, hidden_dim, output_dim)
        self.prototypes = SwaVPrototypes(output_dim, n_prototypes=n_prototypes)

        # enable sinkhorn_gather_distributed to gather features from all gpus
        # while running the sinkhorn algorithm in the loss calculation
        self.criterion = SwaVLoss(sinkhorn_gather_distributed=True)

    def forward(self, x):
        """Calculate prototypes."""
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p

    def training_step(self, batch, batch_idx):
        """Perform training step."""
        self.prototypes.normalize()
        crops, _, _ = batch
        multi_crop_features = [self.forward(x.to(self.device)) for x in crops]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = self.criterion(high_resolution, low_resolution)
        self.log("loss/train", loss)
        return loss

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer],
        list[
            Union[
                torch.optim.lr_scheduler._LRScheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ]
        ],
    ]:
        """Configure optimizers."""
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        return [optimizer], [scheduler]

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
