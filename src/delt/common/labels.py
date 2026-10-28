"""Module for label utils."""

import torch


def labels_to_indices(
    truths: torch.Tensor | list,
    label_names: list[str] | None,
) -> torch.Tensor | list:
    """
    Convert label names in `truths` to label indices, i.e., positions in `label_names`.

    Indices, tensors (hard or soft truths) and lists without label names are returned as they are.

    Args:
        truths (torch.Tensor | list): reference labels, as label names and/or label indices.
        label_names (list[str] | None): names of the labels, in the same order as the label embeddings.

    Returns:
        torch.Tensor | list: `truths` with label names replaced by label indices.

    """
    if not isinstance(truths, (list, tuple)):
        return truths

    if not any(isinstance(truth, str) for truth in truths):
        return truths

    if label_names is None:
        msg = "Cannot use label names in `truths` without `label_names`."
        raise ValueError(msg)

    indices = []
    for truth in truths:
        if isinstance(truth, str):
            if truth not in label_names:
                msg = (
                    f"Unknown label {truth!r}, expected one of {label_names}."
                )
                raise ValueError(msg)
            truth = label_names.index(truth)
        indices.append(truth)
    return indices
