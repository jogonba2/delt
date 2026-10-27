"""Module for evaluating label tuning models."""

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from delt.common import labels_to_indices
from delt.predict import predict_proba


def evaluate(
    input_embeddings: torch.Tensor | list,
    label_embeddings: torch.Tensor | list,
    logit_scale: torch.Tensor,
    truths: torch.Tensor | list,
    label_names: list[str] | None = None,
    batch_size: int = 16,
) -> dict:
    """
    Compute classification metrics on a held-out set.

    Args:
        input_embeddings (torch.Tensor | list): the input embeddings with shape (N, d).
        label_embeddings (torch.Tensor | list): the label embeddings with shape (L, d).
        logit_scale (torch.Tensor): the logit scale in log space with shape (1,), i.e., the logits are multiplied by `logit_scale.exp()`.
        truths (torch.Tensor | list): reference labels as integers (or label names, if `label_names` is set) with shape (N,) or expected probabilities with shape (N, L).
        label_names (list[str] | None): the names of the labels, in the same order as `label_embeddings`.
        batch_size (int): the batch size for inference.

    Returns:
        dict: number of samples, accuracy, macro and weighted F1, and precision, recall, F1 and support per label.

    """
    if isinstance(input_embeddings, list):
        input_embeddings = torch.tensor(input_embeddings)
    if isinstance(label_embeddings, list):
        label_embeddings = torch.tensor(label_embeddings)

    truth_indices = torch.as_tensor(labels_to_indices(truths, label_names))
    if truth_indices.dim() > 1:
        truth_indices = truth_indices.argmax(-1)
    y_true = truth_indices.cpu().numpy()

    probas = predict_proba(
        input_embeddings, label_embeddings, logit_scale, batch_size
    )
    preds = probas.argmax(-1)

    indices = list(range(label_embeddings.shape[0]))
    names = label_names if label_names is not None else map(str, indices)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, preds, labels=indices, zero_division=0
    )
    return {
        "num_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, preds)),
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
