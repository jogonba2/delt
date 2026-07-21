"""Large Multimodal Model teacher for audio classification."""

import torch
from torch.nn import functional as F

from delt.common import base64_encode, generate_completions, make_output_model
from delt.teachers.base import Teacher
from delt.types import Audio


class LMMAudioTeacher(Teacher):
    """
    Large Multimodal Model-based teacher for audio classification.

    Generates structured predictions from audio inputs using LiteLLM.
    """

    def __init__(
        self, model_name: str, decoding_args: dict, system_prompt: str
    ) -> None:
        """
        Initialize an LMM teacher for audio modality.

        Args:
            model_name (str): name of an LMM supporting audio as input modality.
            decoding_args (dict): decoding args supported by LiteLLM.
            system_prompt (str): system prompt to classify with the LMM.

        """
        super().__init__(model_name=model_name)
        self.decoding_args = decoding_args
        self.system_prompt = system_prompt

    def predict_proba(
        self,
        input_data: list[Audio],
        label_set: list[str],
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Compute probabilities from a teacher model for the `input_data` over the `label_set`.

        Each audio in `input_data` must be encoded as `bytes` and only `.wav` format is
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
            "Additionally, only `.wav` format is supported."
        )
        response_format = make_output_model(label_set)
        b64_audios = [base64_encode(example) for example in input_data]

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.system_prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": b64_audio,
                                "format": "wav",
                            },
                        },
                    ],
                }
            ]
            for b64_audio in b64_audios
        ]

        responses = generate_completions(
            self.model_name,
            conversations,
            response_format,
            self.decoding_args,
            batch_size,
        )

        return F.one_hot(
            torch.tensor(
                [label_set.index(output.label) for output in responses]
            ),
            num_classes=len(label_set),
        )
