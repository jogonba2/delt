"""Example script for the audio pipeline."""

import requests

from delt import AudioPipeline


# Define a helper function to download audios
def download_audio(url: str) -> bytes:
    """Download an audio from a direct HTTP(S) URL and return it as bytes."""
    return requests.get(url).content


# Instantiate the pipeline
label_verbalizations = {"bell": "bell", "morse": "morse"}
prompt_template = "That sound is a {}"
encoder_class = "clap"
encoder_name = "laion/larger_clap_music_and_speech"
pipeline = AudioPipeline(
    encoder_name, encoder_class, label_verbalizations, prompt_template
)

# Zero-shot prediction
test_audios = [
    download_audio("https://soundcamp.org/sounds/382/RECEPTION_BELL_oVn.wav"),
    download_audio(
        "https://soundcamp.org/sounds/382/morse_code_oscillator_-_low_pitched_NzH.wav"
    ),
]
preds = pipeline.predict(test_audios, batch_size=8)

# Label-tuning training
train_audios = [
    download_audio(
        "https://soundcamp.org/sounds/382/BELL_TOLL_WITH_BARK_u78.wav"
    ),
    download_audio(
        "https://soundcamp.org/sounds/382/morse_code_oscillator_-_medium_pitched_PTv.wav"
    ),
]
train_truths = ["bell", "morse"]
training_output = pipeline.fit(train_audios, train_truths)

# Calibrate
calibration_audios = [
    download_audio(
        "https://soundcamp.org/sounds/382/BELL_AND_CAR_PASSING_BY_brO.wav"
    ),
    download_audio(
        "https://soundcamp.org/sounds/382/morse_code_oscillator_-_high_pitched_K29.wav"
    ),
]
calibration_truths = ["bell", "morse"]
calibrator = pipeline.calibrate(calibration_audios, calibration_truths)

# Prediction after training
preds = pipeline.predict(test_audios, batch_size=8)

# Evaluate
test_truths = ["bell", "morse"]
results = pipeline.evaluate(test_audios, test_truths, batch_size=8)

# Save the pipeline
pipeline.save("checkpoints/my-audio-model")

# Load the pipeline and predict again
pipeline = AudioPipeline.load("checkpoints/my-audio-model")
preds = pipeline.predict(test_audios, batch_size=8)
