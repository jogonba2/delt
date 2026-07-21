"""Example script for label tuning training."""

import torch

from delt.predict import predict
from delt.train import label_tuning

# Define your input embeddings
input_embeddings = torch.vstack(
    [torch.randn(100, 16), torch.randn(100, 16) + 2]
)

# Define your label embeddings
label_embeddings = torch.vstack([torch.randn(1, 16), torch.randn(1, 16) + 2])

# Define your truth labels
truths = [0] * 100 + [1] * 100

# Zero-shot prediction
preds = predict(input_embeddings, label_embeddings, torch.tensor([1.0]))

# Train the label embeddings
training_output = label_tuning(
    input_embeddings,
    label_embeddings,
    truths,
    epochs=500,
    apply_class_weights=False,
    learning_rate=1e-2,
    dropout=1e-2,
    drift_coefficient=1e-2,
)

# Predict
preds = predict(
    input_embeddings,
    training_output.label_embeddings,
    training_output.logit_scale,
)
