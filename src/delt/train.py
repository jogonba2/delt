"""Module for label tuning training."""

from collections import Counter
from time import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from delt.common import dict_cartesian_product, unimodal_kfold
from delt.predict import predict
from delt.types import TrainingOutput

GRID_SEARCH_PARAMS = {
    "learning_rate": [1e-2, 1e-1],
    "epochs": [1000, 2000],
    "drift_coefficient": [1e-2, 1e-1],
    "dropout": [1e-2, 1e-1],
    "apply_class_weights": [True, False],
}


def compute_class_weights(truths: torch.Tensor) -> torch.Tensor:
    """
    Compute class weights for class weighting training.

    Implements max(c) / c_i for all c_i.

    Args:
        truths (torch.Tensor): reference labels as integers with shape (N,).

    Returns:
        torch.Tensor: where each position i contains the weight of the label i

    """
    counts = torch.bincount(truths)
    return counts.max().float() / counts.float()


def label_tuning(
    input_embeddings: torch.Tensor | list,
    label_embeddings: torch.Tensor | list,
    truths: torch.Tensor | list,
    epochs: int = 2000,
    apply_class_weights: bool = False,
    learning_rate: float = 1e-2,
    dropout: float = 1e-2,
    drift_coefficient: float = 1e-2,
) -> TrainingOutput:
    """
    Run label tuning given a fixed set of hyperparams.

    Args:
        input_embeddings (torch.Tensor | list): the input embeddings with shape (N, d)
        label_embeddings (torch.Tensor | list): the label embeddings with shape (N, d).
        truths (torch.Tensor | list): reference labels as integers with shape (N,) or expected probabilities with shape (N, classes).
        epochs (int): training epochs.
        apply_class_weights (bool): whether to use class weighting or not.
        learning_rate (float): the learning rate.
        dropout (float): dropout applied to label embeddings during training.
        drift_coefficient (float): regularization to preserve how much the tuned label embedings can diverge from the initial ones.

    Returns:
        TrainingOutput: containing the tuned label embeddings (N, d), logit scale (1,), training time, and other outputs for inspecting training.

    """
    # Ensure we deal with tensors
    if isinstance(input_embeddings, list):
        input_embeddings = torch.tensor(input_embeddings)

    if isinstance(label_embeddings, list):
        label_embeddings = torch.tensor(label_embeddings)

    if isinstance(truths, list):
        truths = torch.tensor(truths)

    # Move everything to the same device
    truths = truths.to(input_embeddings.device)
    label_embeddings = label_embeddings.to(input_embeddings.device)

    # Prepare parameters to train
    trainable_label_embeddings = nn.Parameter(
        label_embeddings.clone(), requires_grad=True
    )
    logit_scale = nn.Parameter(
        torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
    )

    # Instantiate optimizer and loss
    optimizer = torch.optim.AdamW(
        params=[trainable_label_embeddings, logit_scale], lr=learning_rate
    )

    # Check whether to apply class weights
    if apply_class_weights:
        assert len(truths.shape) == 1, (
            "Class weights can only be computed with categorical truths."
        )

    loss_fn = nn.CrossEntropyLoss(
        weight=compute_class_weights(truths)
        if apply_class_weights and len(truths.shape) == 1
        else None
    )

    pbar = tqdm(range(epochs), desc="Training")
    start_time = time()
    for _ in pbar:
        drop_label_embeddings = F.dropout(
            trainable_label_embeddings, p=dropout, training=True
        )
        drop_norm_label_embeddings = F.normalize(
            drop_label_embeddings, p=2, dim=1
        )

        logits = (
            input_embeddings @ drop_norm_label_embeddings.T
        ) * logit_scale.exp()

        norm_label_embeddings = F.normalize(
            trainable_label_embeddings, p=2, dim=1
        )
        drift_loss = (
            ((norm_label_embeddings - label_embeddings) ** 2).sum(dim=1).mean()
        )

        classification_loss = loss_fn(logits, truths)
        total_loss = classification_loss + (drift_coefficient * drift_loss)

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix(
            loss=f"{total_loss.item():.4f}",
            cls=f"{classification_loss.item():.4f}",
            drift=f"{drift_loss.item():.4f}",
        )
    end_time = time()
    label_embeddings = norm_label_embeddings.detach()
    logit_scale = logit_scale.detach()

    return TrainingOutput(
        label_embeddings=label_embeddings,
        logit_scale=logit_scale,
        training_time=end_time - start_time,
        loss=total_loss.item(),
        trained_param_count=label_embeddings.numel(),
        hyperparams={
            "epochs": epochs,
            "apply_class_weights": apply_class_weights,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "drift_coefficient": drift_coefficient,
        },
    )


def label_tuning_cv(
    input_embeddings: torch.Tensor | list,
    label_embeddings: torch.Tensor | list,
    truths: torch.Tensor | list,
    num_splits: int = 4,
) -> TrainingOutput:
    """
    Run label tuning with cross validation.

    Determines the best hyper-params through k-fold cv.

    Args:
        input_embeddings (torch.Tensor | list): the input embeddings with shape (N, d).
        label_embeddings (torch.Tensor | list): the label embeddings with shape (N, d).
        truths (torch.Tensor | list): reference labels as integers with shape (N,) or expected probabilities with shape (N, classes).
        num_splits (int): K folds.

    Returns:
        TrainingOutput: containing the tuned label embeddings (N, d), logit scale (1,), training time, and other outputs for monitoring training.

    """
    n_samples = len(truths)
    min_class_size = min(Counter(truths).values())
    n_splits = min(num_splits, n_samples, min_class_size)

    if n_splits < 2:
        raise ValueError(
            f"Cannot perform stratified {n_splits}-fold CV: "
            f"{n_samples=} and {min_class_size=}."
        )

    hyperparams = dict_cartesian_product(GRID_SEARCH_PARAMS)
    train_folds, val_folds = unimodal_kfold(
        input_embeddings, truths, n_splits=n_splits
    )

    best_params, best_score = None, -1

    pbar = tqdm(hyperparams, desc="Hyperparameter search")

    for experiment_params in pbar:
        scores = []

        for train_fold, val_fold in zip(train_folds, val_folds):
            tuned_label_embeddings, logit_scale = label_tuning(
                train_fold["x"],
                label_embeddings,
                train_fold["y"],
                **experiment_params,
            )

            preds = predict(val_fold["x"], tuned_label_embeddings, logit_scale)
            preds = [pred.label for pred in preds]

            scores.append(f1_score(val_fold["y"], preds, average="macro"))

        score = float(np.mean(scores))

        # update tqdm bar with current config score
        pbar.set_postfix(
            macro_f1=f"{score:.4f}",
            best=f"{best_score:.4f}" if best_score >= 0 else "None",
        )

        if score > best_score:
            best_score = score
            best_params = experiment_params

    print("\nBest macro F1:", best_score)
    print("Best params:", best_params)

    # Train final model with best params
    return label_tuning(
        input_embeddings, label_embeddings, truths, **best_params
    )
