"""Example script for the image pipeline with LMM distillation."""

import PIL
import requests

from delt.pipelines import ImagePipeline
from delt.teachers import LMMImageTeacher

# Define your input data and label set
cat_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
dog_url = "https://pngfre.com/wp-content/uploads/dog-poster.png"
images = [
    PIL.Image.open(requests.get(url, stream=True).raw).convert("RGB")
    for url in [cat_url, dog_url]
]
label_set = ["cat", "dog"]

# Instante the teacher model
teacher = LMMImageTeacher(
    "gemini/gemini-3.5-flash",
    {"temperature": 0},
    "Classify the following images into cat or dog.",
)

# Generate the labels (`predict`) or probs (`predict_proba`)
truths = teacher.predict(images, label_set)

# Create the label verbalizations (same order as the label set)
label_verbalizations = {
    "cat": "a cat",
    "dog": "a dog",
}

# Set the encoder
prompt_template = "The animal in the image is a {}"
encoder_class = "siglip"
encoder_name = "google/siglip2-base-patch16-naflex"

# Instantiate the pipeline
pipeline = ImagePipeline(
    encoder_name, encoder_class, label_verbalizations, prompt_template
)

# Zero-shot prediction
preds = pipeline.predict(images, batch_size=8)

# Label-tuning to train the student model
training_output = pipeline.fit(images, truths)

# Prediction after training
preds = pipeline.predict(images, batch_size=8)
