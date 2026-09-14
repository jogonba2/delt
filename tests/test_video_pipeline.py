"""Tests for `delt.VideoPipeline`."""

import random

import pytest
import requests
import torch

from delt import VideoPipeline


@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(13)
    torch.manual_seed(13)
    yield


def download_video(url: str) -> bytes:
    """Download a video from a direct HTTP(S) URL and return it as bytes."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


@pytest.fixture
def video_dataset():
    """Build a tiny video classification deepfake dataset."""
    videos = [
        download_video(
            "https://huggingface.co/datasets/dfb-data/deep-fake-detection-cropped/"
            "resolve/4850ffa259634a34ef92bb741d79baddc22cfb47/"
            "1/DFDC_Dataset/Fake/aaaoqepxnf.mp4"
        ),
        download_video(
            "https://huggingface.co/datasets/dfb-data/deep-fake-detection-cropped/"
            "resolve/4850ffa259634a34ef92bb741d79baddc22cfb47/"
            "1/DFDC_Dataset/Real/ykofirxynw.mp4"
        ),
    ]
    truths = [0, 1]
    return videos, truths


class TestVideoPipeline:
    """Class to test `VideoPipeline`."""

    @classmethod
    def setup_class(cls):
        """Create a `TestVideoPipeline` instance."""
        encoder_class = "xclip"
        encoder_name = "microsoft/xclip-base-patch16-zero-shot"
        label_verbalizations = {
            "fake": "fake",
            "real": "real",
        }
        prompt_template = "The video shows a {} face."
        cls.pipeline = VideoPipeline(
            encoder_name,
            encoder_class,
            label_verbalizations,
            prompt_template,
        )

    def test_label_tuning_improves_zero_shot(self, video_dataset):
        """Check whether label tuning improves or matches zero-shot performance."""
        videos, truths = video_dataset
        zs_preds = torch.as_tensor(
            [pred.label for pred in self.pipeline.predict(videos)]
        )
        self.pipeline.fit(videos, truths)
        lt_preds = torch.as_tensor(
            [pred.label for pred in self.pipeline.predict(videos)]
        )
        truths = torch.as_tensor(truths)
        zs_accuracy = (zs_preds == truths).float().mean().item()
        lt_accuracy = (lt_preds == truths).float().mean().item()
        assert lt_accuracy >= zs_accuracy
