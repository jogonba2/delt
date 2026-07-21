"""Example script for the audio pipeline with LMM distillation."""

import requests

from delt.pipelines import AudioPipeline
from delt.teachers import LMMAudioTeacher

# Define your input data and label set
bell = requests.get(
    "https://soundcamp.org/sounds/382/RECEPTION_BELL_oVn.wav"
).content
morse = requests.get(
    "https://soundcamp.org/sounds/382/morse_code_oscillator_-_medium_pitched_PTv.wav"
).content
audios = [bell, morse]
label_set = ["bell", "morse"]

# Instante the teacher model
teacher = LMMAudioTeacher(
    "gemini/gemini-3.5-flash",
    {"temperature": 0},
    "Classify the following texts into bell or morse",
)

# Generate the labels (`predict`) or probs (`predict_proba`)
truths = teacher.predict(audios, label_set)

# Create the label verbalizations (same order as the label set)
label_verbalizations = {
    "bell": "a bell",
    "morse": "morse code",
}

# Set the encoder
prompt_template = "That sound is a {}"
encoder_class = "clap"
encoder_name = "laion/larger_clap_music_and_speech"

# Instantiate the pipeline
pipeline = AudioPipeline(
    encoder_name, encoder_class, label_verbalizations, prompt_template
)

# Zero-shot prediction
preds = pipeline.predict(audios, batch_size=8)

# Label-tuning to train the student model
training_output = pipeline.fit(audios, truths)

# Prediction after training
preds = pipeline.predict(audios, batch_size=8)
