"""Example script for the video pipeline."""

import subprocess

from delt import VideoPipeline


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
    download_youtube_video("https://www.youtube.com/watch?v=R2mO_XFiXKc"),
    download_youtube_video("https://www.youtube.com/watch?v=LpNVf8sczqU"),
]

label_verbalizations = {
    "programming": "programming course",
    "anime": "anime tv show",
    "french song": "french song",
}
truths = [1, 2]
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
