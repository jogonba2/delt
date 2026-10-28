"""Example script for the image pipeline."""

import PIL
import requests

from delt import ImagePipeline


# Define a helper function to download images
def download_image(url: str) -> PIL.Image.Image:
    """Download an image from a direct HTTP(S) URL and return it PIL Image."""
    return PIL.Image.open(requests.get(url, stream=True).raw).convert("RGB")


# Instantiate the pipeline
label_verbalizations = {"cat": "cat", "dog": "dog"}

prompt_template = "The animal in the image is a {}"
encoder_class = "siglip"
encoder_name = "google/siglip2-base-patch16-naflex"
pipeline = ImagePipeline(
    encoder_name, encoder_class, label_verbalizations, prompt_template
)

# Zero-shot prediction
test_images = [
    download_image("http://images.cocodataset.org/val2017/000000039769.jpg"),
    download_image("https://pngfre.com/wp-content/uploads/dog-poster.png"),
]
preds = pipeline.predict(test_images, batch_size=8)

# Label-tuning training
train_images = [
    download_image(
        "https://images.pexels.com/photos/32557420/pexels-photo-32557420.jpeg"
    ),
    download_image(
        "https://images.pexels.com/photos/30074125/pexels-photo-30074125.jpeg"
    ),
]
train_truths = ["cat", "dog"]
training_output = pipeline.fit(train_images, train_truths)

# Calibrate
calibration_images = [
    download_image(
        "https://marketplace.canva.com/xHPJk/MAHA8XxHPJk/1/tl/canva-MAHA8XxHPJk.jpg"
    ),
    download_image(
        "https://static.vecteezy.com/system/resources/thumbnails/048/775/730/small/"
        "cute-shih-tzu-dog-in-pink-sweatshirt-and-sunglasses-on-pastel-background-photo.jpg"
    ),
]
calibrate_truths = ["cat", "dog"]
calibrator = pipeline.calibrate(calibration_images, calibrate_truths)

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
