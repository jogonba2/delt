"""Module for inference with dual encoder models."""

import numpy as np
import torch

from delt.calibrators import Calibrator
from delt.common import batchify_tensor
from delt.types import DEFAULT_LOGIT_SCALE, Prediction


def predict_proba(
    input_embeddings: torch.Tensor,
    label_embeddings: torch.Tensor,
    logit_scale: torch.Tensor | None = None,
    batch_size: int = 16,
    calibrator: Calibrator | None = None,
) -> np.ndarray:
    """
    Compute the probabilities of each input for each label.

    Args:
        input_embeddings (torch.Tensor): the input embeddings with shape (N, d).
        label_embeddings (torch.Tensor): the label embeddings with shape (L, d).
        logit_scale (torch.Tensor | None): the logit scale in log space with shape (1,), i.e., the logits are multiplied by `logit_scale.exp()`.
                                           If `None`, `DEFAULT_LOGIT_SCALE` is used.
        batch_size (int): the batch size.
        calibrator (Calibrator | None): a fitted calibrator to apply to the probabilities. If `None`, probabilities are not calibrated.

    Returns:
        np.ndarray: array with shape (N, L), with the probabilities of each input for each label.

    """
    if logit_scale is None:
        logit_scale = torch.tensor([DEFAULT_LOGIT_SCALE])

    # Move to the same device
    if label_embeddings.device != input_embeddings.device:
        label_embeddings = label_embeddings.to(input_embeddings)
    if logit_scale.device != input_embeddings.device:
        logit_scale = logit_scale.to(input_embeddings)

    logits = []
    for batch in batchify_tensor(input_embeddings, batch_size):
        with torch.inference_mode():
            logits.append((batch @ label_embeddings.T) * logit_scale.exp())

    probs = torch.vstack(logits).softmax(-1).detach().cpu().numpy()
    return calibrator.calibrate(probs) if calibrator is not None else probs


def predict(
    input_embeddings: torch.Tensor,
    label_embeddings: torch.Tensor,
    logit_scale: torch.Tensor | None = None,
    batch_size: int = 16,
    label_names: list[str] | None = None,
    calibrator: Calibrator | None = None,
) -> list[Prediction]:
    """
    Compute the max prob label and probs per label for each input.

    Args:
        input_embeddings (torch.Tensor): the input embeddings with shape (N, d).
        label_embeddings (torch.Tensor): the label embeddings with shape (L, d).
        logit_scale (torch.Tensor | None): the logit scale in log space with shape (1,), i.e., the logits are multiplied by `logit_scale.exp()`. If `None`, `DEFAULT_LOGIT_SCALE` is used.
        batch_size (int): the batch size.
        label_names (list[str] | None): the names of the labels, in the same order as `label_embeddings`. If `None`, labels are returned as indices.
        calibrator (Calibrator | None): a fitted calibrator to apply to the probabilities. If `None`, probabilities are not calibrated.

    Returns:
        list[Prediction]: list with predictions for each sample, containing the label with max prob (name or index) and probs per label.

    """
    probas = predict_proba(
        input_embeddings, label_embeddings, logit_scale, batch_size, calibrator
    )
    preds = probas.argmax(-1)
    return [
        Prediction(
            probs=probs,
            label=label_names[label.item()]
            if label_names is not None
            else label.item(),
        )
        for probs, label in zip(probas, preds)
    ]
