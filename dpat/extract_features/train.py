import lightning.pytorch as pl

from dpat.data.datasets import PMCHHGImageDataModule
from dpat.extract_features.models import SwAV


def train():
    """Main contrastive learning CLI."""

    model = SwAV(
        lr=0.0001, input_dim=512, hidden_dim=256, output_dim=64, n_prototypes=16
    )

    datamodule = PMCHHGImageDataModule(
        model="swav",
        root_dir=r"D:\Pediatric brain tumours\images-tif",
        train_img_paths_and_targets=r"D:\Pediatric brain tumours\images-tif\splits\medulloblastoma+pilocytic-astrocytoma_pmc-hhg_train-subfold-0-fold-0.csv",
        val_img_paths_and_targets=r"D:\Pediatric brain tumours\images-tif\splits\medulloblastoma+pilocytic-astrocytoma_pmc-hhg_val-subfold-0-fold-0.csv",
        test_img_paths_and_targets=r"D:\Pediatric brain tumours\images-tif\splits\medulloblastoma+pilocytic-astrocytoma_pmc-hhg_test-subfold-0-fold-0.csv",
        mpp=0.2,
        tile_size_x=224,
        tile_size_y=224,
        tile_overlap_x=0,
        tile_overlap_y=0,
        tile_mode="overflow",
        crop=False,
        mask_factory="no_mask",
        mask_foreground_threshold=None,
        mask_root_dir=r"D:\Pediatric brain tumours\images-tif\masks",
        num_workers=0,
        batch_size=4,
        transform=None,
    )
    datamodule.setup("fit")

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        strategy="ddp",
        sync_batchnorm=True,
        fast_dev_run=False,
    )

    trainer.fit(model=model, train_dataloaders=datamodule.train_dataloader())
