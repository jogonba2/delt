"""Large Language Model teacher for text classification."""

import torch
from torch.nn import functional as F

from delt.common import generate_completions, make_output_model
from delt.teachers.base import Teacher


class LLMTextTeacher(Teacher):
    """
    LLM-based teacher for video classification.

    Generates structured predictions from video inputs using LiteLLM.
    """

    def __init__(
        self, model_name: str, decoding_args: dict, system_prompt: str
    ) -> None:
        """
        Initialize an LLM teacher for text modality.

        Args:
            model_name (str): name of an LLM supporting text as input modality.
            decoding_args (dict): decoding args supported by LiteLLM.
            system_prompt (str): system prompt to classify with the LLM.

        """
        super().__init__(model_name=model_name)
        self.decoding_args = decoding_args
        self.system_prompt = system_prompt

    def predict_proba(
        self, input_data: list[str], label_set: list[str], batch_size: int = 8
    ) -> torch.Tensor:
        """
        Compute probabilities from a teacher model for the `input_data` over the `label_set`.

        Args:
            input_data (list[T]): list of input data.
            label_set (list[str]): set of labels in your task.
            batch_size (int): batch size.

        Returns:
            torch.Tensor: output probabilities of shape (N, L).

        """
        response_format = make_output_model(label_set)
        conversations = [
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": example},
            ]
            for example in input_data
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
