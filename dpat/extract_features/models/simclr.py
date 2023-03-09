"""Provide SimCLR model."""
from typing import Union

import lightning.pytorch as pl
import torch
import torchvision
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import nn

from dpat.types import VisionBackbone


class SimCLR(pl.LightningModule):
    """Simple contrastive learning of visual representations [1].

    Depends on Lightly.

    [1] https://arxiv.org/abs/2002.05709
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.CosineAnnealingLR,  # type: ignore # noqa: E501
    ):
        """Build SimCLR model.

        Parameters
        ----------
        backbone : str, default=resnet18
            Backbone to pretrain. Can be any attribute of `torchvision.models`. E.g.
            `resnet9|18`. or `shufflenet_v2_x1_0`.
        optimizer : `OptimizerCallable`, default=torch.optim.Adam
            Optimizer to use.
        scheduler : `SchedulerCallable`, default=CosineAnnealingLR
            Scheduler to use.
        """
        super().__init__()

        self.example_input_array = torch.Tensor(32, 3, 224, 224)

        self.optimizer = optimizer
        self.scheduler = scheduler

        # Get any vision model from torchvision as backbone.
        vision_backbone: VisionBackbone = getattr(torchvision.models, backbone)()

        self.backbone = nn.Sequential(*list(vision_backbone.children())[:-1])

        hidden_dim = vision_backbone.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        # Enable gather_distributed to gather features from all gpus
        # before calculating the loss.
        self.criterion = NTXentLoss(gather_distributed=True)

    def forward(self, x):
        """Calculate prototypes."""
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_idx):
        """Perform training step."""
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
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
