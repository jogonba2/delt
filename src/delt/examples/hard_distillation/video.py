"""Example script for the video pipeline with LMM distillation."""

import subprocess

from delt.pipelines import VideoPipeline
from delt.teachers import LMMVideoTeacher


def download_youtube_video(url: str) -> bytes:
    """
    Download a video from Youtube.

    Args:
        url (str): the url of the video.

    Returns:
        bytes: the video as bytes.

    """
    result = subprocess.run(
        [
            "yt-dlp",
            "-f",
            "worst[ext=mp4]",
            "-o",
            "-",
            url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout


# Set your data and configure the encoder
videos = [
    download_youtube_video("https://www.youtube.com/watch?v=tPEE9ZwTmy0"),
    download_youtube_video("https://www.youtube.com/watch?v=BsIRoA99bBY"),
]

label_set = ["cat", "dog", "person"]


# Instante the teacher model
teacher = LMMVideoTeacher(
    "gemini/gemini-3.5-flash",
    {"temperature": 0},
    "Classify the following texts into bell or morse",
)

# Generate the labels (`predict`) or probs (`predict_proba`)
truths = teacher.predict(videos, label_set)

# Create the label verbalizations (same order as the label set)
label_verbalizations = {
    "cat": "a cat",
    "dog": "a dog",
    "person": "a person",
}
prompt_template = "The video shows a {}"
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
