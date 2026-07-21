"""Module for inference with dual encoder models."""

import numpy as np
import torch

from delt.common import batchify_tensor
from delt.types import Prediction


def predict_proba(
    input_embeddings: torch.Tensor,
    label_embeddings: torch.Tensor,
    logit_scale: torch.Tensor,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Compute the probabilities of each input for each label.

    Args:
        input_embeddings (torch.Tensor): the input embeddings with shape (N, d).
        label_embeddings (torch.Tensor): the label embeddings with shape (L, d).
        logit_scale (torch.Tensor): the logit scale with shape (1,).
        batch_size (int): the batch size.

    Returns:
        np.ndarray: array with shape (N, L), with the probabilities of each input for each label.

    """
    # Move to the same device
    if label_embeddings.device != input_embeddings.device:
        label_embeddings = label_embeddings.to(input_embeddings)
    if logit_scale.device != input_embeddings.device:
        logit_scale = logit_scale.to(input_embeddings)

    logits = []
    for batch in batchify_tensor(input_embeddings, batch_size):
        with torch.inference_mode():
            logits.append((batch @ label_embeddings.T) * logit_scale.exp())

    return torch.vstack(logits).softmax(-1).detach().cpu().numpy()


def predict(
    input_embeddings: torch.Tensor,
    label_embeddings: torch.Tensor,
    logit_scale: torch.Tensor,
    batch_size: int = 16,
) -> list[Prediction]:
    """
    Compute the max prob label and probs per label for each input.

    Args:
        input_embeddings (torch.Tensor): the input embeddings with shape (N, d).
        label_embeddings (torch.Tensor): the label embeddings with shape (L, d).
        logit_scale (torch.Tensor): the logit scale with shape (1,).
        batch_size (int): the batch size.

    Returns:
        list[Prediction]: list with predictions for each sample, containing the label with max prob and probs per label.

    """
    probas = predict_proba(
        input_embeddings, label_embeddings, logit_scale, batch_size
    )
    preds = probas.argmax(-1)
    return [
        Prediction(probs=probs, label=label.item())
        for probs, label in zip(probas, preds)
    ]
