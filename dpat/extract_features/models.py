import lightning.pytorch as pl
import torch
import torchvision
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from torch import nn


class SwaV(pl.LightningModule):
    def __init__(self, lr, input_dim, hidden_dim, output_dim, n_prototypes):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr

        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SwaVProjectionHead(input_dim, hidden_dim, output_dim)
        self.prototypes = SwaVPrototypes(output_dim, n_prototypes=n_prototypes)

        # enable sinkhorn_gather_distributed to gather features from all gpus
        # while running the sinkhorn algorithm in the loss calculation
        self.criterion = SwaVLoss(sinkhorn_gather_distributed=True)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p

    def training_step(self, batch, batch_idx):
        self.prototypes.normalize()
        crops, _, _ = batch
        multi_crop_features = [self.forward(x.to(self.device)) for x in crops]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = self.criterion(high_resolution, low_resolution)
        loss = 10
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim
