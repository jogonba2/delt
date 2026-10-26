"""Example script for the video pipeline with LMM distillation."""

import requests

from delt.pipelines import VideoPipeline
from delt.teachers import LMMVideoTeacher


def download_video(url: str) -> bytes:
    """Download a video from a direct HTTP(S) URL and return it as bytes."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


# Set your data and configure the encoder
videos = [
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

label_set = ["fake", "real"]

# Instantiate the teacher model
teacher = LMMVideoTeacher(
    "gemini/gemini-3.5-flash",
    {"temperature": 0},
    "Classify the following videos into fake or real.",
)

# Generate the labels (`predict`) or probs (`predict_proba`)
truths = teacher.predict(videos, label_set)

# Create the label verbalizations (same order as the label set)
label_verbalizations = {
    "fake": "fake",
    "real": "real",
}
prompt_template = "The video shows a {} face."
encoder_class = "xclip"
encoder_name = "microsoft/xclip-base-patch16-zero-shot"

# Instantiate the pipeline
pipeline = VideoPipeline(
    encoder_name, encoder_class, label_verbalizations, prompt_template
)

# Zero-shot prediction
preds = pipeline.predict(videos, batch_size=8)

# Label-tuning training
training_output = pipeline.fit(videos, truths)

# Prediction after training
preds = pipeline.predict(videos, batch_size=8)
