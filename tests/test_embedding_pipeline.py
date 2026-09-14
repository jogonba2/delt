"""Tests for `delt.EmbeddingPipeline`."""

import random

import pytest
import torch

from delt import EmbeddingPipeline
from delt.types import TrainingOutput


@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(13)
    torch.manual_seed(13)
    yield


@pytest.fixture(params=["two_class", "three_class"])
def embeddings_data(request):
    """Build two datasets where classes are linearly separable."""
    dim = 16
    n_per_class = 32

    # Orthogonal classes
    if request.param == "two_class":
        centers = torch.zeros(2, dim)
        centers[0, 0] = 1.0
        centers[1, 1] = 1.0
    else:
        centers = torch.zeros(3, dim)
        centers[0, 0] = 1.0
        centers[1, 1] = 1.0
        centers[2, 2] = 1.0

    noise = 1e-3

    embeddings = torch.cat(
        [center + noise * torch.randn(n_per_class, dim) for center in centers]
    )

    labels = torch.cat(
        [
            torch.full((n_per_class,), i, dtype=torch.long)
            for i in range(len(centers))
        ]
    )

    label_embeddings = centers.clone()

    return embeddings, labels, label_embeddings


def test_zero_shot_output_length(embeddings_data):
    """Check that the shape of predictions match the shape of the inputs."""
    embeddings, _, label_embeddings = embeddings_data
    pipeline = EmbeddingPipeline(label_embeddings)
    preds = pipeline.predict(embeddings, batch_size=8)
    assert len(preds) == len(embeddings)


def test_zero_shot_valid_labels(embeddings_data):
    """Check that zero-shot is returning valid labels."""
    embeddings, _, label_embeddings = embeddings_data
    pipeline = EmbeddingPipeline(label_embeddings)
    preds = pipeline.predict(embeddings, batch_size=8)
    assert min([pred.label for pred in preds]) >= 0
    assert max([pred.label for pred in preds]) < label_embeddings.shape[0]


def test_label_tuning_return(embeddings_data):
    """Check `training_output` result from `fit` is as expected."""
    embeddings, labels, label_embeddings = embeddings_data
    pipeline = EmbeddingPipeline(label_embeddings)
    training_output = pipeline.fit(embeddings, labels)
    assert training_output is not None
    assert isinstance(training_output, TrainingOutput)


@pytest.mark.skip(reason="Slow for tests.")
def test_label_tuning_cv_return(embeddings_data):
    """Check `training_output` result from `fit` is as expected when `do_cv=True`."""
    embeddings, labels, label_embeddings = embeddings_data
    pipeline = EmbeddingPipeline(label_embeddings)
    training_output = pipeline.fit(embeddings, labels, do_cv=True)
    assert training_output is not None
    assert isinstance(training_output, TrainingOutput)


def test_label_tuning_accuracy(embeddings_data):
    """Check label tuning gives perfect accuracy on the fixture."""
    embeddings, labels, label_embeddings = embeddings_data
    pipeline = EmbeddingPipeline(label_embeddings)
    pipeline.fit(embeddings, labels)
    preds = pipeline.predict(embeddings, batch_size=8)
    pred_labels = torch.as_tensor([pred.label for pred in preds])
    accuracy = (pred_labels == labels).float().mean().item()
    assert accuracy > 0.99


def test_label_tuning_consistency_across_batch_sizes(embeddings_data):
    """Check that predictions do not differ when using different batch sizes."""
    embeddings, labels, label_embeddings = embeddings_data
    pipeline = EmbeddingPipeline(label_embeddings)
    pipeline.fit(embeddings, labels)

    smaller_batch = torch.as_tensor(
        [pred.label for pred in pipeline.predict(embeddings, batch_size=4)]
    )
    bigger_batch = torch.as_tensor(
        [
            pred.label
            for pred in pipeline.predict(
                embeddings, batch_size=len(embeddings)
            )
        ]
    )
    assert torch.equal(smaller_batch, bigger_batch)


def test_label_tuning_continuous_fit(embeddings_data):
    """Check that label embeddings can be trained sequentially."""
    embeddings, labels, label_embeddings = embeddings_data
    pipeline = EmbeddingPipeline(label_embeddings)
    pipeline.fit(embeddings, labels)
    pipeline.fit(embeddings, labels)
    predictions = pipeline.predict(embeddings, batch_size=8)
    assert len(predictions) == len(embeddings)
