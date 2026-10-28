"""Module for calibrator utils."""

import numpy as np
import torch


def to_targets(truths: torch.Tensor | list, num_labels: int) -> np.ndarray:
    """
    Convert reference labels to expected probabilities.

    Args:
        truths (torch.Tensor | list): reference labels as integers with shape (N,) or expected probabilities with shape (N, L).
        num_labels (int): the number of labels, L.

    Returns:
        np.ndarray: expected probabilities with shape (N, L), being one-hot for hard labels.

    """
    truths = torch.as_tensor(truths)
    if truths.dim() == 1:
        truths = torch.nn.functional.one_hot(
            truths.long(), num_classes=num_labels
        )
    if truths.dim() != 2 or truths.shape[-1] != num_labels:
        msg = (
            f"Expected `truths` with shape (N,) or (N, {num_labels}), "
            f"got {tuple(truths.shape)}."
        )
        raise ValueError(msg)
    return truths.detach().cpu().double().numpy()
