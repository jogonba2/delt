"""Tests for `delt.ImagePipeline`."""

import random

import numpy as np
import PIL
import pytest
import requests
import torch

from delt import ImagePipeline


@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(13)
    torch.manual_seed(13)
    yield


@pytest.fixture
def image_dataset():
    """Build a tiny, cat vs. dog dataset."""
    cat_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    dog_url = "https://pngfre.com/wp-content/uploads/dog-poster.png"
    images = [
        PIL.Image.open(requests.get(url, stream=True).raw).convert("RGB")
        for url in [cat_url, dog_url]
    ]
    truths = ["cat", "dog"]
    return images, truths


class TestImagePipeline:
    """Class to test `ImagePipeline`."""

    @classmethod
    def setup_class(cls):
        """Create a `TestImagePipeline` instance."""
        encoder_class = "siglip"
        encoder_name = "google/siglip2-base-patch16-naflex"
        label_verbalizations = {"cat": "cat", "dog": "dog"}
        prompt_template = "The animal in the image is a {}"
        cls.pipeline = ImagePipeline(
            encoder_name,
            encoder_class,
            label_verbalizations,
            prompt_template,
        )

    def test_label_tuning_improves_zero_shot(self, image_dataset):
        """Check whether label tuning improves or matches zero-shot performance."""
        images, truths = image_dataset
        zs_preds = [pred.label for pred in self.pipeline.predict(images)]
        self.pipeline.fit(images, truths)
        lt_preds = [pred.label for pred in self.pipeline.predict(images)]
        zs_accuracy = np.mean([p == t for p, t in zip(zs_preds, truths)])
        lt_accuracy = np.mean([p == t for p, t in zip(lt_preds, truths)])
        assert lt_accuracy >= zs_accuracy
