import torchvision
from torch import nn
from torch import optim
import torch
from torch.functional import F
import lightning.pytorch as pl

class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams["temperature"] > 0.0, "The temperature must be a positive float!"
        # Base model f(.) 
        # TODO: SWAP OUT RESNET FOR SHUFFLENETV2
        self.convnet = torchvision.models.resnet18(
            pretrained=False, num_classes=4 * hidden_dim
        )  # num_classes is the output size of the last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams["max_epochs"], eta_min=self.hparams["lr"] / 50
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        loss = self.loss_fn(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss_fn(batch)
        self.log("train_loss", loss)