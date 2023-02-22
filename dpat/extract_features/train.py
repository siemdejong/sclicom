import lightning.pytorch as pl

from dpat.data.datasets import PMCHHGImageDataModule
from dpat.extract_features.models import SwaV


class PMCHHGTrainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = SwaV(
            lr=0.0001, input_dim=256, hidden_dim=256, output_dim=64, n_prototypes=16
        )

        self.datamodule = PMCHHGImageDataModule(
            model=self.model.__class__.__name__,
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
            num_workers=3,
            batch_size=32,
            transform=None,
        )

    def fit(self) -> None:
        super().fit(model=self.model, datamodule=self.datamodule)


def train():
    """Main contrastive learning CLI."""

    trainer = PMCHHGTrainer(
        max_epochs=10,
        accelerator="gpu",
        # strategy="ddp",
        # sync_batchnorm=True,
        # fast_dev_run=True,
    )

    trainer.fit()


# def cli_main():
#     """Main contrastive learning CLI.

#     Do contrastive learning on a dataset. Parts of this function are copied from [1].

#     References
#     ----------
#     [1] `pl_bolts.models.self_supervised.simclr.simclr_module.cli_main`
#         https://github.com/Lightning-AI/lightning-bolts
#     """
#     from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
#     from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform

#     from models import HHGSimCLR
#     import dpat
#     dpat.install_windows(r"D:\apps\vips-dev-8.14\bin")
#     from dpat.data.datasets import PMCHHGImageDataModule

#     parser = argparse.ArgumentParser()

#     # model args
#     parser: argparse.ArgumentParser = HHGSimCLR.add_model_specific_args(parser)
#     args = parser.parse_args()

#     if args.dataset == "pmc-hhg":
#         normalization = None # pmchhg_normalization()

#         args.gaussian_blur = True
#         args.jitter_strength = 1.0

#         args.batch_size = 16
#         args.max_epochs = 800
#         args.max_steps = 800

#         args.optimizer = "adam"
#         args.learning_rate = 4.8
#         args.final_lr = 0.0048
#         args.start_lr = 0.3
#         args.online_ft = True

#         with open(args.config_file, "r") as stream:
#             try:
#                 conf = yaml.safe_load(stream)
#             except yaml.YAMLError as exc:
#                 print(exc)

#         dm = PMCHHGImageDataModule(**conf)

#         args.input_height = dm.tile_size_x
#         args.num_samples = 8205
#     else:
#         raise NotImplementedError("other datasets have not been implemented till now")

#     dm.train_transforms = SimCLRTrainDataTransform(
#         input_height=args.input_height,
#         gaussian_blur=args.gaussian_blur,
#         jitter_strength=args.jitter_strength,
#         normalize=normalization,
#     )

#     dm.val_transforms = SimCLREvalDataTransform(
#         input_height=args.input_height,
#         gaussian_blur=args.gaussian_blur,
#         jitter_strength=args.jitter_strength,
#         normalize=normalization,
#     )

#     model = HHGSimCLR(**args.__dict__)

#     online_evaluator = None
#     if args.online_ft:
#         # online eval
#         online_evaluator = SSLOnlineEvaluator(
#             drop_p=0.0,
#             hidden_dim=None,
#             z_dim=args.hidden_mlp,
#             num_classes=2,#dm.num_classes,
#             dataset=args.dataset,
#         )

#     lr_monitor = LearningRateMonitor(logging_interval="step")
#     model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss")
#     callbacks = [model_checkpoint, online_evaluator] if args.online_ft else [model_checkpoint]
#     callbacks.append(lr_monitor)

#     trainer = Trainer(
#         max_epochs=args.max_epochs,
#         max_steps=None if args.max_steps == -1 else args.max_steps,
#         gpus=args.gpus,
#         num_nodes=args.num_nodes,
#         accelerator="ddp" if args.gpus > 1 else None,
#         sync_batchnorm=True if args.gpus > 1 else False,
#         precision=32 if args.fp32 else 16,
#         callbacks=callbacks,
#         fast_dev_run=args.fast_dev_run,
#     )

#     trainer.fit(model, datamodule=dm)

# if __name__ == "__main__":
#     cli_main()
