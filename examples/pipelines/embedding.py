"""Example script for the embedding pipeline using precomputed input and label embeddings."""

import torch

from delt import EmbeddingPipeline

# Instantiate the pipeline
label_embeddings = torch.vstack(
    [torch.randn(1, 16) - 3, torch.randn(1, 16) + 3]
)
pipeline = EmbeddingPipeline(
    label_embeddings, label_names=["negative", "positive"]
)

# Zero-shot prediction
test_embeddings = torch.vstack(
    [torch.randn(2, 16) - 3, torch.randn(2, 16) + 3]
)
preds = pipeline.predict(test_embeddings, batch_size=8)

# Label-tuning training
training_embeddings = torch.vstack(
    [torch.randn(5, 16) - 3, torch.randn(5, 16) + 3]
)
training_truths = ["negative"] * 5 + ["positive"] * 5
training_output = pipeline.fit(
    training_embeddings, training_truths, do_cv=False
)

# Predict after training
preds = pipeline.predict(test_embeddings, batch_size=8)

# Calibrate the probabilities with held-out data
calibration_embeddings = torch.vstack(
    [torch.randn(5, 16) - 3, torch.randn(5, 16) + 3]
)
calibration_truths = ["negative"] * 5 + ["positive"] * 5
calibrator = pipeline.calibrate(
    calibration_embeddings, calibration_truths, method="beta"
)

# Predict after calibration
preds = pipeline.predict(test_embeddings, batch_size=8)

# Evaluate
test_truths = ["negative"] * 2 + ["positive"] * 2
results = pipeline.evaluate(test_embeddings, test_truths, batch_size=8)

# Save the pipeline
pipeline.save("checkpoints/my-embedding-model")

# Load the pipeline and predict again
pipeline = EmbeddingPipeline.load("checkpoints/my-embedding-model")
preds = pipeline.predict(test_embeddings, batch_size=8)
