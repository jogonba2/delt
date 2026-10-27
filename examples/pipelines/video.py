"""Example script for the video pipeline."""

import requests

from delt import VideoPipeline


# Define a helper function
def download_video(url: str) -> bytes:
    """Download a video from a direct HTTP(S) URL and return it as bytes."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


# Instantiate the pipeline
label_verbalizations = {
    "fake": "fake",
    "real": "real",
}
prompt_template = "The video shows a {} face."
encoder_class = "xclip"
encoder_name = "microsoft/xclip-base-patch16-zero-shot"
pipeline = VideoPipeline(
    encoder_name, encoder_class, label_verbalizations, prompt_template
)

# Zero-shot prediction
test_videos = [
    download_video(
        "https://huggingface.co/datasets/dfb-data/deep-fake-detection-cropped/"
        "resolve/4850ffa259634a34ef92bb741d79baddc22cfb47/"
        "1/DFDC_Dataset/Fake/aaaoqepxnf.mp4"
    ),
    download_video(
        "https://huggingface.co/datasets/dfb-data/deep-fake-detection-cropped/"
        "resolve/4850ffa259634a34ef92bb741d79baddc22cfb47/"
        "1/DFDC_Dataset/Real/ykofirxynw.mp4"
    ),
]
preds = pipeline.predict(test_videos, batch_size=8)

# Label-tuning training
train_videos = [
    download_video(
        "https://huggingface.co/datasets/dfb-data/deep-fake-detection-cropped/"
        "resolve/4850ffa259634a34ef92bb741d79baddc22cfb47/1/DFDC_Dataset/Real/ylntmnbfvk.mp4"
    ),
    download_video(
        "https://huggingface.co/datasets/dfb-data/deep-fake-detection-cropped/"
        "resolve/4850ffa259634a34ef92bb741d79baddc22cfb47/1/DFDC_Dataset/Fake/aawhvztjdt.mp4"
    ),
]
train_truths = ["real", "fake"]
training_output = pipeline.fit(train_videos, train_truths)

# Prediction after training
preds = pipeline.predict(test_videos, batch_size=8)

# Evaluate
test_truths = ["fake", "real"]
results = pipeline.evaluate(test_videos, test_truths, batch_size=8)

# Save the pipeline
pipeline.save("checkpoints/my-video-model")

# Load the pipeline and predict again
pipeline = VideoPipeline.load("checkpoints/my-video-model")
preds = pipeline.predict(test_videos, batch_size=8)
