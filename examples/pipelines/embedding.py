"""Example script for the embedding pipeline using precomputed input and label embeddings."""

import torch

from delt import EmbeddingPipeline

# Define your input embeddings
input_embeddings = torch.vstack([torch.randn(5, 16), torch.randn(5, 16) + 5])

# Define your label embeddings
label_embeddings = torch.vstack([torch.randn(1, 16), torch.randn(1, 16) + 5])

# Define your truth labels
truths = ["positive"] * 5 + ["negative"] * 5

# Instantiate the pipeline
pipeline = EmbeddingPipeline(
    label_embeddings, label_names=["negative", "positive"]
)

# Zero-shot prediction
preds = pipeline.predict(input_embeddings, batch_size=8)

# Label-tuning training
training_output = pipeline.fit(input_embeddings, truths, do_cv=False)

# Prediction after training
preds = pipeline.predict(input_embeddings, batch_size=8)
print(preds)
