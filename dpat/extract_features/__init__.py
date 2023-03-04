"""Extract feature vectors from the tiles."""

import platform

if platform.system() != "Linux":
    import dpat

    dpat.install_windows(r"D:\apps\vips-dev-8.14\bin")

# import os
# import urllib.request
# from copy import deepcopy
# from urllib.error import HTTPError

# import lightning.pytorch as pl
# import matplotlib
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.utils.data as data
# import torchvision
# from torchvision import transforms
# from tqdm import tqdm
# from train import train_simclr
# from transformations import ContrastiveTransformations


# def main():
#     # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
#     DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
#     # Path to the folder where the pretrained models are saved
#     CHECKPOINT_PATH = os.environ.get(
#         "PATH_CHECKPOINT", "saved_models/ContrastiveLearning/"
#     )
#     # In this notebook, we use data loaders with heavier computational processing.
#     # It is recommended to use as many
#     # workers as possible in a data loader, which corresponds to the number of CPU
#     # cores
#     NUM_WORKERS = os.cpu_count()

#     # Setting the seed
#     pl.seed_everything(42)

#     # Ensure that all operations are deterministic on GPU (if used) for
#     # reproducibility
#     torch.backends.cudnn.determinstic = False
#     torch.backends.cudnn.benchmark = False

#     device = (
#         torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#     )

#     # Create checkpoint path if it doesn't exist yet
#     os.makedirs(CHECKPOINT_PATH, exist_ok=True)

#     # TODO: CREATE TILE DATASETS

#     simclr_model = train_simclr(
#         batch_size=256,
#         hidden_dim=128,
#         lr=5e-4,
#         temperature=0.07,
#         weight_decay=1e-4,
#         max_epochs=500,
#     )


# if __name__ == "__main__":
#     main()
