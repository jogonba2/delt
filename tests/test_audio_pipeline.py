"""Tests for `delt.AudioPipeline`."""

import random

import pytest
import requests
import torch

from delt import AudioPipeline


@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(13)
    torch.manual_seed(13)
    yield


@pytest.fixture
def sounds_dataset():
    """Build a tiny, bell vs. morse dataset."""
    bell = requests.get(
        "https://soundcamp.org/sounds/382/RECEPTION_BELL_oVn.wav"
    ).content
    morse = requests.get(
        "https://soundcamp.org/sounds/382/morse_code_oscillator_-_medium_pitched_PTv.wav"
    ).content
    audios = [bell, morse]
    truths = [0, 1]
    return audios, truths


class TestAudioPipeline:
    """Class to test `AudioPipeline`."""

    @classmethod
    def setup_class(cls):
        """Create a `TestAudioPipeline` instance."""
        encoder_class = "clap"
        encoder_name = "laion/larger_clap_music_and_speech"
        label_verbalizations = {"bell": "bell", "morse": "morse"}
        prompt_template = "That sound is a {}"
        cls.pipeline = AudioPipeline(
            encoder_name,
            encoder_class,
            label_verbalizations,
            prompt_template,
        )

    def test_label_tuning_improves_zero_shot(self, sounds_dataset):
        """Check whether label tuning improves or matches zero-shot performance."""
        audios, truths = sounds_dataset
        zs_preds = torch.as_tensor(
            [pred.label for pred in self.pipeline.predict(audios)]
        )
        self.pipeline.fit(audios, truths)
        lt_preds = torch.as_tensor(
            [pred.label for pred in self.pipeline.predict(audios)]
        )
        truths = torch.as_tensor(truths)
        zs_accuracy = (zs_preds == truths).float().mean().item()
        lt_accuracy = (lt_preds == truths).float().mean().item()
        assert lt_accuracy >= zs_accuracy
