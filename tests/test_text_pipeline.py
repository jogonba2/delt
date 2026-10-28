"""Tests for `delt.TextPipeline`."""

import random

import numpy as np
import pytest
import torch

from delt import TextPipeline


@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(13)
    torch.manual_seed(13)
    yield


@pytest.fixture
def text_dataset():
    """Build a tiny, obviously-polarized sentiment dataset."""
    texts = [
        "I love this movie, it was fantastic!",
        "This is the worst film I have ever seen.",
        "What a wonderful, joyful experience.",
        "I hated every second of it, truly terrible.",
        "Amazing, I would watch it again!",
        "Awful, boring, and a complete waste of time.",
    ]
    truths = [
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
    ]
    return texts, truths


class TestTextPipeline:
    """Class to test `TextPipeline`."""

    @classmethod
    def setup_class(cls):
        """Create a `TestTextPipeline` instance."""
        encoder_class = "sentence-transformer"
        encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
        label_verbalizations = {
            "positive": "really positive",
            "negative": "really negative",
        }
        prompt_template = "This text is {}"
        cls.pipeline = TextPipeline(
            encoder_name,
            encoder_class,
            label_verbalizations,
            prompt_template,
        )

    def test_label_tuning_improves_zero_shot(self, text_dataset):
        """Check whether label tuning improves or matches zero-shot performance."""
        texts, truths = text_dataset
        zs_preds = [pred.label for pred in self.pipeline.predict(texts)]
        self.pipeline.fit(texts, truths)
        lt_preds = [pred.label for pred in self.pipeline.predict(texts)]
        zs_accuracy = np.mean([p == t for p, t in zip(zs_preds, truths)])
        lt_accuracy = np.mean([p == t for p, t in zip(lt_preds, truths)])
        assert lt_accuracy >= zs_accuracy
