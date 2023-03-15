import pathlib

import numpy as np
import pandas as pd
import pytest

from dpat.exceptions import DpatOutputDirectoryExistsError
from dpat.splits.create_splits import create_splits, file_of_paths_to_list
from dpat.splits.create_splits import test as splits_test

from .common import image_files


@pytest.fixture(scope="session")
def paths_file(tmp_path_factory, image_files) -> pathlib.Path:
    """Create a csv file with a path in every row."""
    df = pd.DataFrame(image_files)
    fn = tmp_path_factory.mktemp("data") / f"paths.csv"
    df.to_csv(fn)

    return fn


@pytest.fixture(scope="session")
def path_to_labels_file(tmp_path_factory, image_files) -> pathlib.Path:
    """Create a csv file with a path in every row with their diagnoses.

    Case ids must be prefixed with PMC_HHG_
    """
    diagnoses = []
    case_ids = []
    for i, _ in enumerate(image_files):
        # Force both diagnoses to occur.
        diagnosis = "a" if i < 0.5 * len(image_files) else "b"
        diagnoses.append(diagnosis)
        case_ids.append(f"PMC_HHG_{i}")

    data = np.array([case_ids, diagnoses]).T
    df = pd.DataFrame(data, columns=["case_id", "diagnosis"])

    fn = tmp_path_factory.mktemp("data") / f"labels.csv"
    df.to_csv(fn, index=False)

    return fn


class TestCreateSplits:
    """Test Create Splits."""

    def test_file_of_paths_to_list(self, paths_file):
        """Test if paths can be read from file with paths per row."""
        pathlist = file_of_paths_to_list(paths_file)
        assert isinstance(pathlist, list)
        assert len(pathlist)

    @pytest.mark.parametrize("num_subfolds", [1, 5])
    def test_create_splits(self, num_subfolds, image_files, path_to_labels_file):
        dataset_name = "test"
        create_splits(
            image_dir=image_files[0].parent,
            path_to_labels_file=path_to_labels_file,
            include_pattern=["*.bmp"],
            dataset_name=dataset_name,
            overwrite=True,
            num_subfolds=num_subfolds,
        )

        # Reuse tests written that are ran at runtime anyway,
        # because we want to be sure.
        splits_test(
            path_to_labels_file=path_to_labels_file,
            save_to_dir=image_files[0].parent / "splits",
            dataset_name=dataset_name,
            diagnoses_fn="a+b",
            num_subfolds=num_subfolds,
        )

    def test_raise_dir_exists_error(self, image_files, path_to_labels_file):
        with pytest.raises(DpatOutputDirectoryExistsError):
            create_splits(
                image_dir=image_files[0].parent,
                path_to_labels_file=path_to_labels_file,
                include_pattern=["*bmp"],
                dataset_name="test",
            )

    def test_filter_diagnosis(self, image_files, path_to_labels_file):
        dataset_name = "test"
        num_subfolds = 1
        create_splits(
            image_dir=image_files[0].parent,
            path_to_labels_file=path_to_labels_file,
            include_pattern=["*.bmp"],
            dataset_name=dataset_name,
            filter_diagnosis=["a"],
            overwrite=True,
            num_subfolds=num_subfolds,
        )
        splits_test(
            path_to_labels_file=path_to_labels_file,
            save_to_dir=image_files[0].parent / "splits",
            dataset_name=dataset_name,
            diagnoses_fn="a",
            num_subfolds=num_subfolds,
        )
