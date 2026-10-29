"""Tests for `delt.train`."""

import random

import pytest
import torch
from torch.nn import functional as F

from delt.train import label_tuning


@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(13)
    torch.manual_seed(13)
    yield


@pytest.fixture
def separable():
    """Build a linearly separable dataset with three orthogonal classes."""
    dim, n_per_class = 16, 32
    centers = torch.zeros(3, dim)
    centers[0, 0] = centers[1, 1] = centers[2, 2] = 1.0
    embeddings = torch.cat(
        [c + 1e-3 * torch.randn(n_per_class, dim) for c in centers]
    )
    labels = torch.arange(3).repeat_interleave(n_per_class)
    return embeddings, labels, centers


def test_returned_embeddings_include_the_last_step(separable):
    """Check that the returned embeddings are the ones after the last step."""
    embeddings, labels, centers = separable
    output = label_tuning(
        embeddings, centers, labels, epochs=1, learning_rate=1e-1
    )
    assert not torch.allclose(
        output.label_embeddings, F.normalize(centers, dim=1)
    )


def test_initial_embeddings_are_normalized(separable):
    """Check that the scale of the initial embeddings does not matter."""
    embeddings, labels, centers = separable
    outputs = []
    for scale in (1.0, 3.0):
        torch.manual_seed(0)
        outputs.append(
            label_tuning(embeddings, scale * centers, labels, epochs=20)
        )
    assert torch.allclose(
        outputs[0].label_embeddings, outputs[1].label_embeddings
    )


def test_logit_scale_stays_in_range(separable):
    """Check that a large learning rate does not drive the scale out of range."""
    from delt.defaults import MAX_LOGIT_SCALE

    embeddings, labels, centers = separable
    output = label_tuning(
        embeddings,
        centers,
        labels,
        epochs=300,
        learning_rate=1.0,
        dropout=1e-2,
    )

    # Without limits, this run ends with a negative logit scale (~ -1.6),
    # i.e., logits scaled by less than 1 and almost uniform probabilities
    assert 0 <= output.logit_scale.item() <= MAX_LOGIT_SCALE
