import glob
import pathlib
from typing import Iterable, Optional

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def file_of_paths_to_list(path: str) -> list[str]:
    """Reads a file with a path per row and returns a list of paths."""
    content: list = []
    with open(path, "r") as f:
        while line := f.readline().rstrip():
            content.append(line)
    return content


def test_overlap(save_to_dir: str, dataset_name) -> None:
    """
    Loads the paths to images of the train-val-test splits as previously produced, and tests
    1. If there are duplicate images within a split;
    2. If there are duplicates between splits;
    If this fails, these splits should not be used.
    """
    save_to_dir = pathlib.Path(save_to_dir)
    for fold in range(5):
        train_slides = file_of_paths_to_list(
            save_to_dir / f"paths_{dataset_name}-train-fold-{fold}.csv"
        )
        val_slides = file_of_paths_to_list(
            save_to_dir / f"paths_{dataset_name}-val-fold-{fold}.csv"
        )
        test_slides = file_of_paths_to_list(
            save_to_dir / f"paths_{dataset_name}-test-fold-{fold}.csv"
        )

        # No duplicates within itself
        assert len(set(train_slides)) == len(train_slides)
        assert len(set(val_slides)) == len(val_slides)
        assert len(set(test_slides)) == len(test_slides)

        # No duplicates with any other set
        assert len(set(train_slides).intersection(set(val_slides))) == 0
        assert len(set(train_slides).intersection(set(test_slides))) == 0
        assert len(set(test_slides).intersection(set(val_slides))) == 0


def test_lengths(save_to_dir: str, dataset_name: str) -> None:
    """Test if the length of the train+val+test is the same length for each fold."""
    lengths = []
    for fold in range(5):
        fold_length = 0
        for subset in ["train", "val", "test"]:
            fold_length += len(
                file_of_paths_to_list(
                    pathlib.Path(
                        f"{save_to_dir}/paths_{dataset_name}-{subset}-fold-{fold}.csv"
                    )
                )
            )
        lengths.append(fold_length)
    assert len(set(lengths)) == 1


def test_distributions(
    path_to_labels_file: str,
    save_to_dir: str,
    dataset_name: str,
    filter_diagnosis: Iterable[str],
) -> None:
    """
    Tests if the fraction of positive binarized classes for both mHRD and tHRD is between 0.45 and 0.55
    """
    path_to_patient_df = pd.read_csv(f"{save_to_dir}/paths_to_patient_id.csv")
    labels_df = pd.read_csv(
        f"{save_to_dir}/{dataset_name}-DeepSMILE_{pathlib.Path(path_to_labels_file).stem}.csv"
    )
    for fold in range(5):
        for subset in ["train", "val", "test"]:
            paths = file_of_paths_to_list(
                pathlib.Path(
                    f"{save_to_dir}/paths_{dataset_name}-{subset}-fold-{fold}.csv"
                )
            )
            patient_ids = path_to_patient_df[path_to_patient_df["paths"].isin(paths)][
                "case_id"
            ]
            subset_labels_df = labels_df[labels_df["case_id"].isin(patient_ids)]

            # TODO: change the upper and lower bound.
            # Small datasets will not be stratified well to the same percentages.
            mean = 1 / len(filter_diagnosis)
            lower_bound = mean - 0.05
            upper_bound = mean + 0.05
            counts = subset_labels_df["diagnosis"].value_counts(normalize=True)
            for diagnosis, count in zip(counts.index, counts.tolist()):
                print(
                    f"Percentage of {diagnosis} in \t{subset} \tfold {fold}: {count:.2f}%"
                )
                # assert lower_bound <= count <= upper_bound
                assert 0 <= count <= 1


def test(
    path_to_labels_file: str,
    save_to_dir: str,
    dataset_name: str,
    filter_diagnosis: Iterable[str],
) -> None:
    # Assert that there's no overlap between train/val, train/test, val/test.
    test_overlap(save_to_dir, dataset_name)

    # Assert that the length of test_i + val_i + train_i are the same for all i
    test_lengths(save_to_dir, dataset_name)

    # Check if the fraction of labels is around 1/N_classes for each fold
    test_distributions(path_to_labels_file, save_to_dir, dataset_name, filter_diagnosis)


def create_splits(
    image_dir: str,
    path_to_labels_file: str,
    dataset_name: str,
    save_to_dir: str = "splits",
    filter_diagnosis: Optional[Iterable[str]] = None,
) -> None:
    """
    Create data splits. The csv label file is loaded.
    5-fold train-test stratified k-fold split are created.
    In every train split, it creates a random train-val split.
    Patients without diagnosis are dropped.
    Patients with diagnosis given in `filter_diagnosis` are used.
    Stratify on diagnosis.

    Also performs tests on the splits.

    Adapted from HISSL [1].

    Parameters
    ----------
    image_dir : str
        Input directory to fetch filenames from.
    path_to_labels_file : str
        Path to the csv file containing "case" and "diagnosis" columns.
    dataset_name : str
        Name to give the splits.
    save_to_dir : str, default="splits"
        Path to save the split files to.
    filter_diagnosis : iterable of str, optional
        Iterable of strings choosing the diagnoses to create the splits for.

    References
    ----------
    [1] https://github.com/NKI-AI/hissl/blob/126d181e31aa66e404a0707532ad9e546097162a/tools/reproduce_deepsmile/4_create_splits_for_tcga_bc/create_splits_tcga_bc.py
    """
    # The directory should not exist, to avoid overwriting previously calculated splits.
    save_to_dir: pathlib.Path = pathlib.Path(save_to_dir)
    save_to_dir.mkdir(exist_ok=True)

    ID_NAME = "case_id"
    df = pd.read_csv(path_to_labels_file)

    paths = pd.DataFrame({"paths": list(glob.glob(f"{image_dir}/*"))})
    paths["case_id"] = paths["paths"].apply(
        lambda path: "_".join(pathlib.Path(path).stem.split("_")[1:4])
    )
    paths["image_id"] = paths["paths"].apply(
        lambda path: "_".join(pathlib.Path(path).stem.split("_")[4:])
    )

    paths.to_csv(save_to_dir / "paths_to_patient_id.csv")

    # Use stratified KFold split for the initial 5 folds
    skf = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )  # split into test and (train)

    # Use random shuffle split for the train-val split
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=0.25, random_state=42
    )  # split into val and train

    # Drop images without diagnosis.
    DROP = ["diagnosis"]
    len_before_nan_drop = len(df)
    df = df.dropna(subset=DROP).reset_index()
    print(
        f"{len_before_nan_drop - len(df)}/{len_before_nan_drop} NaNs of {DROP} are dropped."
    )

    # Filter dataframe by diagnosis
    if filter_diagnosis:
        len_before_filter_drop = len(df)
        df = df[df["diagnosis"].isin(filter_diagnosis)].reset_index()
        print(
            f"{len_before_filter_drop - len(df)}/{len_before_filter_drop} of {DROP} not in {filter_diagnosis} are dropped."
        )
    else:
        filter_diagnosis = df["diagnosis"].unique()

    df["diagnosis_num"] = df["diagnosis"].astype("category").cat.codes

    stratify_on = ["diagnosis"]
    X = df[ID_NAME]
    y = df[stratify_on]

    for fold, (train_index, test_index) in enumerate(skf.split(X=X, y=y)):
        X_trainval = df.iloc[train_index][ID_NAME]  # Get the train-val indices
        y_trainval = df.iloc[train_index][stratify_on]  # Get the train-val indices
        sss.get_n_splits(X=X_trainval, y=y_trainval)  # Split train-val into train & val
        train_index, val_index = next(
            iter(sss.split(X=X_trainval, y=y_trainval))
        )  # Get the indices

        for subfold, subfold_index in zip(
            ["train", "val", "test"], [train_index, val_index, test_index]
        ):
            subfoldname = f"{subfold}-fold-{fold}"  # Set as column name and filename
            df[subfoldname] = 0  # Set all patients to not belong to the new fold

            # The subfold index are the row numbers, so we get the original index
            if subfold in ["train", "val"]:
                indices = X_trainval.index[subfold_index]
            else:
                indices = subfold_index

            # Set those indices as given by the shufflesplit to be 1 in the newly created column
            df.loc[indices, subfoldname] = 1

            # Get the patient IDs that belong to this split
            patients = df[df[subfoldname] == 1][ID_NAME]

            # Get the image paths that belong to these patients
            subfold_slides = paths[paths["case_id"].isin(patients)]["paths"]

            # Save those image paths to a .csv file
            subfold_slides.to_csv(
                save_to_dir / f"paths_{dataset_name}-{subfoldname}.csv",
                header=None,
                index=False,
            )

    # Save the file with labels and splits
    df.to_csv(
        f"{save_to_dir}/{dataset_name}-DeepSMILE_{pathlib.Path(path_to_labels_file).stem}.csv"
    )

    # Run some tests with the recently saved files
    test(path_to_labels_file, save_to_dir, dataset_name, filter_diagnosis)


if __name__ == "__main__":
    create_splits(
        r"D:\Pediatric brain tumours\images-tif",
        r"D:\Pediatric brain tumours\labels.csv",
        "pmc-hhg",
        r"D:\Pediatric brain tumours\splits",
        ["pilocytic astrocytoma", "medulloblastoma"],
    )