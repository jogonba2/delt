"""Module for evaluating label tuning models."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from delt.calibrators import Calibrator
from delt.common import labels_to_indices
from delt.predict import predict_proba


def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> float:
    """
    Compute the expected calibration error (ECE) of the max prob label.

    The samples are grouped in `n_bins` equal-width bins of confidence (the max prob), and the ECE is the average gap between the accuracy and the mean confidence of the bins, weighted by the number of samples in each bin.

    Args:
        probs (np.ndarray): predicted probabilities with shape (N, L).
        y_true (np.ndarray): reference labels as integers with shape (N,).
        n_bins (int): the number of bins.

    Returns:
        float: the ECE, between 0 (perfectly calibrated) and 1.

    """
    confidences = probs.max(-1)
    correct = probs.argmax(-1) == y_true
    # Bin i covers (i / n_bins, (i + 1) / n_bins]
    bins = np.clip(np.ceil(confidences * n_bins).astype(int) - 1, 0, None)
    ece = 0.0
    for i in range(n_bins):
        in_bin = bins == i
        if in_bin.any():
            gap = abs(correct[in_bin].mean() - confidences[in_bin].mean())
            ece += in_bin.mean() * gap
    return float(ece)


def evaluate(
    input_embeddings: torch.Tensor | list | np.ndarray,
    label_embeddings: torch.Tensor | list | np.ndarray,
    logit_scale: torch.Tensor,
    truths: torch.Tensor | list | np.ndarray,
    label_names: list[str] | None = None,
    batch_size: int = 16,
    calibrator: Calibrator | None = None,
) -> dict:
    """
    Compute classification metrics on a held-out set.

    Args:
        input_embeddings (torch.Tensor | list | np.ndarray): the input embeddings with shape (N, d).
        label_embeddings (torch.Tensor | list | np.ndarray): the label embeddings with shape (L, d).
        logit_scale (torch.Tensor): the logit scale in log space with shape (1,), i.e., the logits are multiplied by `logit_scale.exp()`.
        truths (torch.Tensor | list | np.ndarray): reference labels as integers (or label names, if `label_names` is set) with shape (N,) or expected probabilities with shape (N, L).
        label_names (list[str] | None): the names of the labels, in the same order as `label_embeddings`.
        batch_size (int): the batch size for inference.
        calibrator (Calibrator | None): a fitted calibrator to apply to the probabilities. If `None`, probabilities are not calibrated.

    Returns:
        dict: number of samples, accuracy, expected calibration error (`ece`), macro and weighted F1, and precision, recall, F1 and support per label.

    """
    truth_indices = torch.as_tensor(labels_to_indices(truths, label_names))
    if truth_indices.dim() > 1:
        truth_indices = truth_indices.argmax(-1)
    y_true = truth_indices.cpu().numpy()

    probas = predict_proba(
        input_embeddings, label_embeddings, logit_scale, batch_size, calibrator
    )
    preds = probas.argmax(-1)

    indices = list(range(len(label_embeddings)))
    names = label_names if label_names is not None else map(str, indices)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, preds, labels=indices, zero_division=0
    )
    return {
        "num_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, preds)),
        "ece": expected_calibration_error(probas, y_true),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted")),
        "per_label": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in zip(indices, names)
        },
    }
