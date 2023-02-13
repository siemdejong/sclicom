from dlup.data.dataset import TiledROIsSlideImageDataset, ConcatDataset
import pathlib

class PMCHHGTileDataset(ConcatDataset):
    """
    add documentation on how this dataset works
    Args:
        add docstrings for the parameters
    """

    def __init__(self, cfg, path, split, dataset_name, data_source):
        #split should only be train
        #dataset_name should be passed as 'tcga_tile_folder' - name from json
        #dataset_source is passed as ['tile_dataset'] -name (of data source) referring to the dataset class


        #Was used and worked for testing
        # self.cfg = VisslDatasetCatalog.get(dataset_name)  #retrieves the dict with the paths from the catalog
        # split = split.lower()
        # path_images = self.cfg[split][0]
        # path_labels = self.cfg[split][1]

        self.path = pathlib.Path(path)
        super().__init__([TiledROIsSlideImageDataset(fn) for fn in self.path.glob("*")])

    def num_samples(self):
        """
        Size of the dataset
        """
        return self.__len__()

    def __getitem__(self, idx: int):
        """
        implement how to load the data corresponding to idx element in the dataset
        from your data source
        """
        tile_dict = ConcatDataset.__getitem__(self, idx)

        return tile_dict["image"]