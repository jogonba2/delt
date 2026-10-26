"""Example script for the text pipeline with Jev distillation."""

import os

from dotenv import load_dotenv

from delt.pipelines import TextPipeline
from delt.teachers import JevTextTeacher

load_dotenv()

# Define your input data and label set
texts = ["I hate you", "I love you"]
label_set = ["positive", "negative"]

# Instantiate the teacher model (through OpenRouter, although any provider is supported)
teacher = JevTextTeacher(
    "jev-latest",
    "What is the sentiment of this text?",
    "User text: {}",
    base_url="https://openrouter.ai/api",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# Generate the labels (`predict`) or probs (`predict_proba`)
truth_probas = teacher.predict_proba(texts, label_set)

# Create the label verbalizations (same order as the label set)
label_verbalizations = {
    "positive": "really positive",
    "negative": "really negative",
}

# Set the encoder
prompt_template = "This text is {}"
encoder_class = "sentence-transformer"
encoder_name = "sentence-transformers/all-MiniLM-L6-v2"

# Instantiate the pipeline
pipeline = TextPipeline(
    encoder_name, encoder_class, label_verbalizations, prompt_template
)

# Zero-shot prediction
preds = pipeline.predict(texts, batch_size=8)

# Label-tuning to train the student model
training_output = pipeline.fit(texts, truth_probas)

# Prediction after training
preds = pipeline.predict(texts, batch_size=8)
