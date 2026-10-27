"""Example script for the audio pipeline."""

import requests

from delt import AudioPipeline

# Instantiate the pipeline
label_verbalizations = {"bell": "bell", "morse": "morse"}
prompt_template = "That sound is a {}"
encoder_class = "clap"
encoder_name = "laion/larger_clap_music_and_speech"
pipeline = AudioPipeline(
    encoder_name, encoder_class, label_verbalizations, prompt_template
)

# Zero-shot prediction
test_bell = requests.get(
    "https://soundcamp.org/sounds/382/RECEPTION_BELL_oVn.wav"
).content
test_morse = requests.get(
    "https://soundcamp.org/sounds/382/morse_code_oscillator_-_medium_pitched_PTv.wav"
).content
test_audios = [test_bell, test_morse]
preds = pipeline.predict(test_audios, batch_size=8)

# Label-tuning training
train_bell = requests.get(
    "https://soundcamp.org/sounds/382/BELL_TOLL_WITH_BARK_u78.wav"
).content
train_morse = requests.get(
    "https://soundcamp.org/sounds/382/morse_code_oscillator_-_medium_pitched_PTv.wav"
).content
train_audios = [test_morse, test_bell]
train_truths = ["morse", "bell"]
training_output = pipeline.fit(train_audios, train_truths)

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
