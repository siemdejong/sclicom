import lightning.pytorch as pl
from torch.utils.data import DataLoader

class PMCHHGDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(self.PMC_HHG_train, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.PMC_HHG_val, batch_size=64)

    def test_dataloader(self):
        return DataLoader(self.PMC_HHG_test, batch_size=64)