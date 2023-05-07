"""Provide SimCLR model."""
import lightning.pytorch as pl
import torch
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightning.pytorch.cli import OptimizerCallable
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch import nn

from dpat.types import VisionBackbone


class SimCLR(pl.LightningModule):
    """Simple contrastive learning of visual representations [1].

    Depends on Lightly.

    [1] https://arxiv.org/abs/2002.05709
    """

    def __init__(
        self,
        backbone: str = "shufflenetv2_w1",
        pretrained: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
    ):
        """Build SimCLR model.

        Parameters
        ----------
        backbone : str, default=shufflenetv2_w1
            Backbone to pretrain. Can be any model provided by pytorchcv,
            e.g. `resnet18_wd4|shufflenetv2_w1`.
        pretrained : bool, default=False,
            Specify if the model should already be trained on ImageNet.
        optimizer : `OptimizerCallable`, default=torch.optim.Adam
            Optimizer to use.
        scheduler : `SchedulerCallable`, default=CosineAnnealingLR
            Scheduler to use.
        """
        super().__init__()

        self.example_input_array = torch.Tensor(32, 3, 224, 224)

        self.optimizer = optimizer

        # Get any vision model from pytorchcv as backbone.
        vision_backbone: VisionBackbone = ptcv_get_model(
            backbone, pretrained=pretrained
        )

        self.backbone = nn.Sequential(*list(vision_backbone.children())[:-1])

        hidden_dim = vision_backbone.output.in_features
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

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure optimizers."""
        optimizer = self.optimizer(self.parameters())
        return [optimizer]

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: torch.optim.Optimizer
    ) -> None:
        """Set gradients to None instead of zero.

        This improves performance.
        """
        optimizer.zero_grad(set_to_none=True)
