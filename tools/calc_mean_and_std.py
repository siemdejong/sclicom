"""Calculate mean and standard deviation of a dataset."""
import tempfile

from torchvision.transforms import Compose, ToTensor

from dpat.data import PMCHHGImageDataset

# Make a temporary file make the dataset read from.
concatenated_file = tempfile.TemporaryFile()

base_file = (
    "/gpfs/home2/sdejong/pmchhg/images-tif/splits-final"
    "/medulloblastoma+pilocytic-astrocytoma_pmchhg_{fold}-subfold-0-fold-0.csv"
)

read_files = [base_file.format(fold=fold) for fold in ["train", "val", "test"]]

for f in read_files:
    with open(f, "rb") as infile:
        concatenated_file.write(infile.read())

# Go back to the beginning of the file.
concatenated_file.seek(0)

dataset = PMCHHGImageDataset(
    root_dir="/home/sdejong/pmchhg/images-tif",
    image_paths_and_targets=concatenated_file,
    mask_factory="load_from_disk",
    mask_root_dir="/home/sdejong/pmchhg/masks",
    transform=Compose([ToTensor()]),
)
