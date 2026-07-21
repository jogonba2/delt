"""Large Multimodal Model teacher for image classification."""

from io import BytesIO

import torch
from torch.nn import functional as F

from delt.common import base64_encode, generate_completions, make_output_model
from delt.teachers.base import Teacher
from delt.types import Image


class LMMImageTeacher(Teacher):
    """
    Large Multimodal Model-based teacher for image classification.

    Generates structured predictions from image inputs using LiteLLM.
    """

    def __init__(
        self, model_name: str, decoding_args: dict, system_prompt: str
    ) -> None:
        """
        Initialize an LMM teacher for image modality.

        Args:
            model_name (str): name of an LMM supporting image as input modality.
            decoding_args (dict): decoding args supported by LiteLLM.
            system_prompt (str): system prompt to classify with the LMM.

        """
        super().__init__(model_name=model_name)
        self.decoding_args = decoding_args
        self.system_prompt = system_prompt

    def predict_proba(
        self,
        input_data: list[Image],
        label_set: list[str],
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Compute probabilities from a teacher model for the `input_data` over the `label_set`.

        Each image in `input_data` must be wrapped in a `PIL` `Image` and only `png` and `jpeg`
        formats are supported for LLM distillation.

        Args:
            input_data (list[T]): list of input data.
            label_set (list[str]): set of labels in your task.
            batch_size (int): batch size.

        Returns:
            torch.Tensor: output probabilities of shape (N, L).

        """
        response_format = make_output_model(label_set)
        b64_images = []
        for image in input_data:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            b64_images.append(
                base64_encode(data=buffer.getvalue(), mime_type="image/png")
            )

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.system_prompt,
                        },
                        {"type": "image_url", "image_url": {"url": b64_image}},
                    ],
                }
            ]
            for b64_image in b64_images
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
