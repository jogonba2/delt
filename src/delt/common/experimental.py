"""Module for experimental utils."""

from typing import Any

import torch
from sklearn.model_selection import StratifiedKFold


def unimodal_kfold(
    data: list[Any],
    labels: list[str],
    n_splits: int = 4,
    random_state: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Create k folds for unimodal settings.

    Args:
        data (list[Any]): A list of input data samples.
        labels (list[str]): A list of labels corresponding to the input data samples.
        n_splits (int): The number of folds for cross-validation.
        random_state (int): The random seed for reproducibility.

    Returns:
        tuple[list[dict], list[dict]]: A tuple containing two lists of dictionaries for training and validation splits.

    """
    kfold = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    splits = kfold.split(data, labels)
    train_splits, val_splits = [], []
    for train_idxs, val_idxs in splits:
        if isinstance(data, torch.Tensor):
            train_data = data[train_idxs]
            val_data = data[val_idxs]
        else:
            train_data = [data[i] for i in train_idxs]
            val_data = [data[i] for i in val_idxs]

        train_labels = [labels[i] for i in train_idxs]
        val_labels = [labels[i] for i in val_idxs]

        train_splits.append({"x": train_data, "y": train_labels})
        val_splits.append({"x": val_data, "y": val_labels})
    return train_splits, val_splits
