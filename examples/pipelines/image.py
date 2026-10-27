"""Example script for the image pipeline."""

import PIL
import requests

from delt import ImagePipeline

# Instantiate the pipeline
label_verbalizations = {"cat": "cat", "dog": "dog"}

prompt_template = "The animal in the image is a {}"
encoder_class = "siglip"
encoder_name = "google/siglip2-base-patch16-naflex"
pipeline = ImagePipeline(
    encoder_name, encoder_class, label_verbalizations, prompt_template
)

# Zero-shot prediction
test_cat_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
test_dog_url = "https://pngfre.com/wp-content/uploads/dog-poster.png"
test_images = [
    PIL.Image.open(requests.get(url, stream=True).raw).convert("RGB")
    for url in [test_cat_url, test_dog_url]
]
preds = pipeline.predict(test_images, batch_size=8)

# Label-tuning training
train_cat_url = (
    "https://images.pexels.com/photos/32557420/pexels-photo-32557420.jpeg"
)
train_dog_url = (
    "https://images.pexels.com/photos/30074125/pexels-photo-30074125.jpeg"
)
train_images = [
    PIL.Image.open(requests.get(url, stream=True).raw).convert("RGB")
    for url in [train_dog_url, train_cat_url]
]
truths = ["dog", "cat"]
training_output = pipeline.fit(train_images, truths)

# Prediction after training
preds = pipeline.predict(test_images, batch_size=8)

# Evaluate
test_truths = ["cat", "dog"]
results = pipeline.evaluate(test_images, test_truths, batch_size=8)

# Save the pipeline
pipeline.save("checkpoints/my-image-model")

# Load the pipeline and predict again
pipeline = ImagePipeline.load("checkpoints/my-image-model")
preds = pipeline.predict(test_images, batch_size=8)
