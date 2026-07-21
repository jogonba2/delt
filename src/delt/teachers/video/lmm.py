"""Large Multimodal Model teacher for video classification."""

import torch
from torch.nn import functional as F

from delt.common import base64_encode, generate_completions, make_output_model
from delt.teachers.base import Teacher
from delt.types import Video


class LMMVideoTeacher(Teacher):
    """
    Large Multimodal Model-based teacher for video classification.

    Generates structured predictions from video inputs using LiteLLM.
    """

    def __init__(
        self, model_name: str, decoding_args: dict, system_prompt: str
    ) -> None:
        """
        Initialize an LMM teacher for video modality.

        Args:
            model_name (str): name of an LMM supporting video as input modality.
            decoding_args (dict): decoding args supported by LiteLLM.
            system_prompt (str): system prompt to classify with the LMM.

        """
        super().__init__(model_name=model_name)
        self.decoding_args = decoding_args
        self.system_prompt = system_prompt

    def predict_proba(
        self,
        input_data: list[Video],
        label_set: list[str],
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Compute probabilities from a teacher model for the `input_data` over the `label_set`.

        Each image in `input_data` must be encoded as `bytes` and only `mp4` format is
        supported for LLM distillation.

        Args:
            input_data (list[T]): list of input data.
            label_set (list[str]): set of labels in your task.
            batch_size (int): batch size.

        Returns:
            torch.Tensor: output probabilities of shape (N, L).

        """
        assert all(isinstance(example, bytes) for example in input_data), (
            "Audio must be encoded as `bytes` for LLM distillation. "
            "Additionally, only `.mp4` format is supported."
        )

        response_format = make_output_model(label_set)
        b64_videos = [base64_encode(example) for example in input_data]

        conversations = [
            [
                {
                    "role": "user",
                    "content": self.system_prompt,
                }
            ]
            for _ in b64_videos
        ]

        extra_body = [
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": self.system_prompt},
                            {
                                "inline_data": {
                                    "mime_type": "video/mp4",
                                    "data": b64_video,
                                }
                            },
                        ],
                    }
                ]
            }
            for b64_video in b64_videos
        ]

        responses = generate_completions(
            self.model_name,
            conversations,
            response_format,
            self.decoding_args,
            batch_size,
            extra_body,
        )

        return F.one_hot(
            torch.tensor(
                [label_set.index(output.label) for output in responses]
            ),
            num_classes=len(label_set),
        )
