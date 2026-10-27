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


def _named_pipeline(embeddings_data):
    _, _, label_embeddings = embeddings_data
    names = [f"class_{i}" for i in range(label_embeddings.shape[0])]
    return EmbeddingPipeline(label_embeddings, label_names=names), names


def test_predict_returns_label_names(embeddings_data):
    """Check that predicted labels are names when `label_names` is set."""
    embeddings, _, _ = embeddings_data
    pipeline, names = _named_pipeline(embeddings_data)
    preds = pipeline.predict(embeddings, batch_size=8)
    assert all(isinstance(pred.label, str) for pred in preds)
    assert all(pred.label == names[int(pred.probs.argmax())] for pred in preds)


def test_predict_returns_indices_without_label_names(embeddings_data):
    """Check that predicted labels are indices when `label_names` is not set."""
    embeddings, _, label_embeddings = embeddings_data
    preds = EmbeddingPipeline(label_embeddings).predict(embeddings)
    assert all(isinstance(pred.label, int) for pred in preds)


def test_fit_with_label_names_matches_indices(embeddings_data):
    """Check that truths as label names are equivalent to truths as indices."""
    embeddings, labels, _ = embeddings_data
    _, names = _named_pipeline(embeddings_data)
    mixed = [names[i] if j % 2 else i for j, i in enumerate(labels.tolist())]
    outputs = []
    for truths in (labels.tolist(), [names[i] for i in labels], mixed):
        pipeline, _ = _named_pipeline(embeddings_data)
        torch.manual_seed(13)
        outputs.append(pipeline.fit(embeddings, truths))
    for output in outputs[1:]:
        assert torch.equal(
            output.label_embeddings, outputs[0].label_embeddings
        )


def test_fit_with_unknown_label_name(embeddings_data):
    """Check that unknown label names in truths raise an error."""
    embeddings, _, _ = embeddings_data
    pipeline, _ = _named_pipeline(embeddings_data)
    with pytest.raises(ValueError, match="Unknown label"):
        pipeline.fit(embeddings, ["unknown"] * len(embeddings))


def test_fit_with_label_names_requires_label_names(embeddings_data):
    """Check that label names in truths need `label_names` to be set."""
    embeddings, _, label_embeddings = embeddings_data
    pipeline = EmbeddingPipeline(label_embeddings)
    with pytest.raises(ValueError, match="label_names"):
        pipeline.fit(embeddings, ["class_0"] * len(embeddings))
