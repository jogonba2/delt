"""Module with utils for video processing."""

from collections import Counter
from io import BytesIO

import av
import numpy as np

from delt.types import Video


def read_video_pyav(
    container: av.container.InputContainer,
    indices: np.ndarray,
    image_format: str = "rgb24",
) -> np.ndarray:
    """
    Decode and extracts specific frames from a PyAV video container.

    Args:
        container (av.container.InputContainer): a PyAV video container.
        indices (np.ndarray): sorted list or array of target frame indices to extract.
        image_format (str): output color format.

    Returns:
        np.ndarray: Stacked array of video frames with shape (N, H, W, C).

    """
    container.seek(0)
    counts = Counter(indices.tolist())
    end_index = indices[-1]
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        repeat = counts.get(i, 0)
        if repeat:
            arr = frame.to_ndarray(format=image_format)
            frames.extend([arr] * repeat)

    return np.stack(frames)


def sample_frame_indices(
    seg_len: int,
    clip_len: int = 32,
    frame_sample_rate: int = 4,
    seed: int = 13,
):
    """
    Generate a sequence of frame indices via randomized temporal sampling.

    Args:
        clip_len (int): number of frames to sample.
        frame_sample_rate (int): step size between consecutive sampled frames.
        seg_len (int): total frame count of the video segment.
        seed (int): for reproducibility.

    Returns: np.ndarray: array of selected integer frame indices.

    """
    np.random.seed(seed)

    converted_len = clip_len * frame_sample_rate

    # Video too short
    if seg_len <= converted_len:
        indices = np.linspace(0, max(seg_len - 1, 0), clip_len)
        return indices.astype(np.int64)

    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len

    indices = np.linspace(start_idx, end_idx - 1, clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def extract_frames(
    video: Video,
    clip_len: int = 32,
    frame_sample_rate: int = 4,
    image_format: str = "rgb24",
) -> np.ndarray:
    """
    Extract a random chunk from the video.

    Args:
        video (Video): a video.
        clip_len (int): number of frames to output.
        frame_sample_rate (int): sampling rate.
        image_format (str): pixel format for the output.

    Returns:
        np.ndarray: Decoded and stacked frames array of shape (N, H, W, C).

    """
    if isinstance(video, bytes):
        container = av.open(BytesIO(video))
        total_frames = container.streams.video[0].frames
        indices = sample_frame_indices(
            seg_len=total_frames,
            clip_len=clip_len,
            frame_sample_rate=frame_sample_rate,
        )
        return read_video_pyav(container, indices, image_format=image_format)

    else:
        frames = [frame.numpy().transpose(1, 2, 0) for frame in video]
        indices = sample_frame_indices(
            seg_len=len(frames),
            clip_len=clip_len,
            frame_sample_rate=frame_sample_rate,
        )
        return np.stack([frames[i] for i in indices])
