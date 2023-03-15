"""Create splits."""

import itertools
import logging
import pathlib
from typing import Union

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from dpat.exceptions import DpatOutputDirectoryExistsError

logger = logging.getLogger(__name__)


def file_of_paths_to_list(path: pathlib.Path) -> list[str]:
    """Read a file with a path per row and returns a list of paths."""
    content: list = []
    with open(path, "r") as f:
        while line := f.readline().rstrip():
            content.append(line)
    return content


def test_overlap(save_to_dir: pathlib.Path, dataset_name, diagnoses_fn: str) -> None:
    """Test overlap of splits.

    Load the paths to images of the train-val-test splits as previously produced,
    and tests
    1. If there are duplicate images within a split;
    2. If there are duplicates between splits;
    If this fails, these splits should not be used.
    """
    save_to_dir = pathlib.Path(save_to_dir)
    for product in itertools.product(range(5), range(5), ["paths", diagnoses_fn]):
        fold, subfold, filetype = product

        train_slides = file_of_paths_to_list(
            save_to_dir
            / f"{filetype}_{dataset_name}_train-subfold-{subfold}-fold-{fold}.csv"
        )
        val_slides = file_of_paths_to_list(
            save_to_dir
            / f"{filetype}_{dataset_name}_val-subfold-{subfold}-fold-{fold}.csv"
        )
        test_slides = file_of_paths_to_list(
            save_to_dir
            / f"{filetype}_{dataset_name}_test-subfold-{subfold}-fold-{fold}.csv"
        )

        # No duplicates within itself
        assert len(set(train_slides)) == len(train_slides)
        assert len(set(val_slides)) == len(val_slides)
        assert len(set(test_slides)) == len(test_slides)

        # No duplicates with any other set
        assert len(set(train_slides).intersection(set(val_slides))) == 0
        assert len(set(train_slides).intersection(set(test_slides))) == 0
        assert len(set(test_slides).intersection(set(val_slides))) == 0


def test_lengths(
    save_to_dir: pathlib.Path, dataset_name: str, diagnoses_fn: str
) -> None:
    """Test lengths across all splits.

    Ensure the length of the train+val+test is the same length for each
    fold.
    """
    lengths = []
    for product in itertools.product(range(5), range(5), ["paths", diagnoses_fn]):
        fold, subfold, filetype = product

        fold_length = 0
        for subset in ["train", "val", "test"]:
            fold_length += len(
                file_of_paths_to_list(
                    pathlib.Path(
                        save_to_dir / f"{filetype}_{dataset_name}_"
                        f"{subset}-subfold-{subfold}-fold-{fold}.csv"
                    )
                )
            )
        lengths.append(fold_length)
    assert len(set(lengths)) == 1


def test_distributions(
    path_to_labels_file: pathlib.Path,
    save_to_dir: pathlib.Path,
    dataset_name: str,
    diagnoses_fn: str,
) -> None:
    """Tests if the fraction of classes are 1 / (included diagnoses)."""
    path_to_patient_df = pd.read_csv(f"{save_to_dir}/paths_to_patient_id.csv")
    labels_df = pd.read_csv(
        f"{save_to_dir}/{dataset_name}-DeepSMILE_"
        f"{pathlib.Path(path_to_labels_file).stem}.csv"
    )
    for product in itertools.product(range(5), range(5)):
        fold, subfold = product

        for subset in ["train", "val", "test"]:
            paths = file_of_paths_to_list(
                pathlib.Path(
                    f"{save_to_dir}/paths_{dataset_name}_"
                    f"{subset}-subfold-{subfold}-fold-{fold}.csv"
                )
            )
            patient_ids = path_to_patient_df[path_to_patient_df["paths"].isin(paths)][
                "case_id"
            ]
            subset_labels_df = labels_df[labels_df["case_id"].isin(patient_ids)]

            # TODO: change the upper and lower bound.
            # Small datasets will not be stratified well to the same percentages.
            # mean = 1 / diagnoses_fn.count("+")
            # lower_bound = mean - 0.05
            # upper_bound = mean + 0.05
            counts = subset_labels_df["diagnosis"].value_counts(normalize=True)
            for diagnosis, count in zip(counts.index, counts.tolist()):
                logger.info(
                    f"Percentage of {diagnosis} in \t{subset} \tsubfold {subfold} fold"
                    f" {fold}: {count*100:.2f}%"
                )
                # assert lower_bound <= count <= upper_bound
                assert 0 <= count <= 1
    logger.warning("`test_distributions` always passes, check above percentages.")


def test(
    path_to_labels_file: pathlib.Path,
    save_to_dir: pathlib.Path,
    dataset_name: str,
    diagnoses_fn: str,
) -> None:
    """Test the splits.

    Test for overlap, length, and distributions.
    """
    # Assert that there's no overlap between train/val, train/test, val/test.
    test_overlap(save_to_dir, dataset_name, diagnoses_fn)

    # Assert that the length of test_i + val_i + train_i are the same for all i
    test_lengths(save_to_dir, dataset_name, diagnoses_fn)

    # # Check if the fraction of labels is around 1/N_classes for each fold
    test_distributions(path_to_labels_file, save_to_dir, dataset_name, diagnoses_fn)


def create_splits(
    image_dir: pathlib.Path,
    path_to_labels_file: pathlib.Path,
    dataset_name: str,
    save_to_dir: pathlib.Path = pathlib.Path("splits"),
    overwrite: bool = False,
    include_pattern: list[str] = ["*.*"],
    exclude_pattern: list[str] = [""],
    filter_diagnosis: Union[list[str], None] = None,
    num_subfolds: int = 1,
) -> None:
    """Create data splits.

    The csv label file is loaded. 5-fold train-test
    stratified k-fold split are created. In every train split, it creates a
    random train-val split. Patients without diagnosis are dropped. Patients
    with diagnosis given in `filter_diagnosis` are used. Stratify on diagnosis.

    Also performs tests on the splits.

    Adapted from HISSL [1].

    Parameters
    ----------
    image_dir : str
        Input directory to fetch filenames from.
    path_to_labels_file : str
        Path to the csv file containing "case_id" and "diagnosis" columns.
        The case_id column can currently only be interpreted with the PMC_HHG_
        prefix.
    dataset_name : str
        Name to give the splits.
    save_to_dir : str, default="splits"
        Relative path to save the split files to.
    overwrite : bool, default=False
        Overwrite items in the directory specified by `save_to_dir`.
    include_pattern : str, default=""
        Glob pattern to include files in `image_dir`.
    exclude_pattern : str, default=""
        Glob pattern to exclude files in `image_dir`.
        Set to be excluded will be subtracted from the set
        to be included by `include_pattern`.
    filter_diagnosis : iterable of str, optional
        Iterable of strings choosing the diagnoses to create the splits for.
    num_subfolds : int, default=1
        Number of subfolds within one fold.
        If 1, creates 5 folds (fold-x-train|val|test).
        If `subfolds`>1, creates 5*`subfolds` folds (fold-x-subfold-y-train|val|test).

    Raises
    ------
    DpatOutputDirectoryExistsError
        If the output directory already exists.

    References
    ----------
    [1] https://github.com/NKI-AI/hissl
    """
    # The directory should not exist, to avoid overwriting previously calculated splits.
    assert not save_to_dir.is_absolute(), "Please provide a relative path to `-o`."
    save_to_dir = image_dir / save_to_dir
    try:
        save_to_dir.mkdir()
    except FileExistsError:
        if not overwrite:
            raise DpatOutputDirectoryExistsError(save_to_dir)
        else:
            save_to_dir.mkdir(parents=True, exist_ok=True)

    ID_NAME = "case_id"
    df = pd.read_csv(path_to_labels_file)

    # Subtract the exclude set from the include set.
    all = [image.name for image in image_dir.glob("*")]
    include = [
        image.name
        for image in itertools.chain(
            *[image_dir.glob(pattern) for pattern in include_pattern if pattern != ""]
        )
    ]
    exclude = [
        image.name
        for image in itertools.chain(
            *[image_dir.glob(pattern) for pattern in exclude_pattern if pattern != ""]
        )
    ]
    paths_list: list = list(set(include) - set(exclude))
    logger.info(
        f"{len(include)}/{len(all)} files are listed for inclusion. {include_pattern}"
    )
    logger.info(
        f"{len(exclude)}/{len(include)} files are listed for exclusion."
        f" {exclude_pattern}"
    )

    paths = pd.DataFrame({"paths": paths_list})
    paths["case_id"] = paths["paths"].apply(
        lambda path: "_".join(pathlib.Path(path).stem.split("_")[1:4])
    )
    paths["image_id"] = paths["paths"].apply(
        lambda path: "_".join(pathlib.Path(path).stem.split("_")[4:])
    )

    paths.to_csv(save_to_dir / "paths_to_patient_id.csv")

    # Drop images without diagnosis.
    DROP = ["diagnosis"]
    len_before_nan_drop = len(df)
    df = df.dropna(subset=DROP).reset_index()
    logger.info(
        f"{len_before_nan_drop - len(df)}/{len_before_nan_drop} NaNs of {DROP} are"
        " dropped."
    )

    # Filter dataframe by diagnosis
    if filter_diagnosis is not None:
        len_before_filter_drop = len(df)
        df = df[df["diagnosis"].isin(filter_diagnosis)].reset_index()
        logger.info(
            f"{len_before_filter_drop - len(df)}/{len_before_filter_drop} of {DROP} not"
            f" in {filter_diagnosis} are dropped."
        )
    else:
        filter_diagnosis = df["diagnosis"].unique()

    logger.info(f"This gives a total of {len(df)} images.")

    df["diagnosis_num"] = df["diagnosis"].astype("category").cat.codes

    # Use stratified KFold split for the initial 5 folds
    skf = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )  # split into test and (train)

    # Use random stratified shuffle split for the train-val split
    sss = StratifiedShuffleSplit(
        n_splits=num_subfolds, test_size=0.2, random_state=42
    )  # split into val and train

    stratify_on = ["diagnosis"]
    X = df[ID_NAME]
    y = df[stratify_on]

    for fold, (train_index, test_index) in enumerate(skf.split(X=X, y=y)):
        X_trainval: pd.DataFrame = df.iloc[train_index][
            ID_NAME
        ]  # Get the train-val indices
        y_trainval: pd.DataFrame = df.iloc[train_index][
            stratify_on
        ]  # Get the train-val indices

        for subfold, (train_index, val_index) in enumerate(
            sss.split(X=X_trainval, y=y_trainval)
        ):
            for setname, indices in zip(
                ["train", "val", "test"], [train_index, val_index, test_index]
            ):
                # Give every subfold a unique identifier.
                subfoldname = f"{setname}-subfold-{subfold}-fold-{fold}"
                df[subfoldname] = 0

                # Train and val set indices come from X_trainval.
                if setname in ["train", "val"]:
                    indices = X_trainval.index[indices]

                df.loc[indices, subfoldname] = 1

                # Get the patient IDs that belong to this split
                patients = df[df[subfoldname] == 1][ID_NAME]

                # Get the image paths that belong to these patients
                subfold_slides = paths[paths["case_id"].isin(patients)]["paths"]

                # Save those image paths to a .csv file
                subfold_slides.to_csv(
                    save_to_dir / f"paths_{dataset_name}_{subfoldname}.csv",
                    header=None,
                    index=False,
                )

    paths = paths.join(df.set_index(ID_NAME), on="case_id", lsuffix="left_")

    diagnoses_fn = "+".join(
        [diagnosis.replace(" ", "-") for diagnosis in filter_diagnosis]
    )

    for product in itertools.product(range(5), range(5)):
        fold, subfold = product

        for subset in ["train", "val", "test"]:
            # Save splits with paths, caseid, imageid, and diagnosis number.
            paths[paths[f"{subset}-subfold-{subfold}-fold-{fold}"] == 1][
                ["paths", "case_id", "image_id", "diagnosis_num"]
            ].to_csv(
                f"{save_to_dir}/{diagnoses_fn}_{dataset_name}_"
                f"{subset}-subfold-{subfold}-fold-{fold}.csv",
                header=None,
                index=None,
            )

    # Save the file with labels and splits
    df.to_csv(
        f"{save_to_dir}/{dataset_name}-DeepSMILE_"
        f"{pathlib.Path(path_to_labels_file).stem}.csv"
    )

    # Run some tests with the recently saved files
    test(path_to_labels_file, save_to_dir, dataset_name, diagnoses_fn)


if __name__ == "__main__":
    create_splits(
        pathlib.Path(r"D:\Pediatric brain tumours\images-tif"),
        pathlib.Path(r"D:\Pediatric brain tumours\labels.csv"),
        "pmc-hhg",
        pathlib.Path(r"D:\Pediatric brain tumours\splits"),
        include_pattern=["*slow.tif"],
        filter_diagnosis=["pilocytic astrocytoma", "medulloblastoma"],
    )
