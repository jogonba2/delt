"""Example script for the audio pipeline."""

import requests

from delt import AudioPipeline

# Set your data and configure the encoder
bell = requests.get(
    "https://soundcamp.org/sounds/382/RECEPTION_BELL_oVn.wav"
).content
morse = requests.get(
    "https://soundcamp.org/sounds/382/morse_code_oscillator_-_medium_pitched_PTv.wav"
).content
audios = [bell, morse]
label_verbalizations = {"bell": "bell", "morse": "morse"}
truths = [0, 1]
prompt_template = "That sound is a {}"
encoder_class = "clap"
encoder_name = "laion/larger_clap_music_and_speech"

# Instantiate the pipeline
pipeline = AudioPipeline(
    encoder_name, encoder_class, label_verbalizations, prompt_template
)

# Zero-shot prediction
preds = pipeline.predict(audios, batch_size=8)
print(preds)

# Label-tuning training
training_output = pipeline.fit(audios, truths)

# Prediction after training
preds = pipeline.predict(audios, batch_size=8)
print(preds)
