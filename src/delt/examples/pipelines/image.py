"""Example script for the image pipeline."""

import PIL
import requests

from delt import ImagePipeline

# Set your data and configure the encoder
cat_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
dog_url = "https://pngfre.com/wp-content/uploads/dog-poster.png"
images = [
    PIL.Image.open(requests.get(url, stream=True).raw).convert("RGB")
    for url in [cat_url, dog_url]
]
label_verbalizations = {"cat": "cat", "dog": "dog"}
truths = [0, 1]
prompt_template = "The animal in the image is a {}"
encoder_class = "siglip"
encoder_name = "google/siglip2-base-patch16-naflex"

# Instantiate the pipeline
pipeline = ImagePipeline(
    encoder_name, encoder_class, label_verbalizations, prompt_template
)

# Zero-shot prediction
preds = pipeline.predict(images, batch_size=8)

# Label-tuning training
training_output = pipeline.fit(images, truths)

# Prediction after training
preds = pipeline.predict(images, batch_size=8)
