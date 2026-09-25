"""Jev teacher for text classification."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from tqdm import tqdm
from typesafe_sdk import ChoiceAnswer, RetryPolicy, TypeSafeClient

from delt.teachers.base import Teacher


def classify(
    client: TypeSafeClient,
    model_name: str,
    text: str,
    text_template: str,
    instruction: str,
    choices: dict[str, str],
) -> ChoiceAnswer:
    """
    Classify a single text using the Jev `model_name` through the TypeSafeAI `client`.

    Args:
        client (TypeSafeClient): TypeSafeAI client from their SDK.
        model_name (str): name of the Jev model to be used.
        text (str): text to be classified.
        text_template (str): template to format the input texts.
        instruction (str): instruction for the Jev model.
        choices (dict[str, str]): label verbalizations for the Jev model.

    Returns:
        ChoiceAnswer: a choice answer from Jev.

    """
    result = client.system_one(
        model=model_name,
        state=text_template.format(text),
        questions={
            "classification_decision": {
                "type": "choice",
                "instructions": instruction,
                "criteria": choices,
            },
        },
    )
    return result.answers["classification_decision"]


def batch_classify(
    client: TypeSafeClient,
    model_name: str,
    texts: list[str],
    text_template: str,
    instruction: str,
    choices: dict[str, str],
    num_workers: int = 4,
) -> list[ChoiceAnswer]:
    """
    Classify a batch of texts using the Jev `model_name` through the TypeSafeAI `client`.

    Args:
        client (TypeSafeClient): TypeSafeAI client from their SDK.
        model_name (str): name of the Jev model to be used.
        texts (list[str]): input texts to be classified.
        text_template (str): template to format the input texts.
        instruction (str): instruction for the Jev model.
        choices (dict[str, str]): label verbalizations for the Jev model.
        num_workers (int): max num of workers for concurrent requests.

    Returns:
        list[ChoiceAnswer]: a list of choice answers, one for input text, from Jev.

    """
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(
            tqdm(
                executor.map(
                    lambda text: classify(
                        client=client,
                        model_name=model_name,
                        text=text,
                        text_template=text_template,
                        instruction=instruction,
                        choices=choices,
                    ),
                    texts,
                ),
                total=len(texts),
                desc="Classifying with Jev...",
            )
        )


class JevTextTeacher(Teacher):
    """
    Jev-based teacher for text classification.

    Generates structured predictions from text inputs using TypeSafeAI's Jev.
    """

    def __init__(
        self,
        model_name: str,
        instruction: str,
        text_template: str = "{}",
        base_url: str | None = None,
        api_key: str | None = None,
        retry_args: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> None:
        """
        Initialize an Jev teacher for text modality.

        Args:
            model_name (str): name of an Jev model supporting text as input modality.
            instruction (str): instruction for the Jev model.
            text_template (str): template to format the input texts, e.g., "User text: {}".
            base_url (Optional[str]): url to the provider.
            api_key (Optional[str]): api key for the provider.
            retry_args (Optional[dict[str, Any]]): arguments for retries on TypeSafeAI SDK.
            timeout (float): timeout for requests.

        """
        super().__init__(model_name=model_name)

        if retry_args is None:
            retry_args = {
                "max_retries": 3,
                "backoff_max": 0.2,
                "timeout": 1.0,
            }
        self.client = TypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            retry=RetryPolicy(**retry_args),
            timeout=timeout,
        )
        self.instruction = instruction
        self.text_template = text_template

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
        choices = {label: label for label in label_set}
        answers = batch_classify(
            self.client,
            self.model_name,
            input_data,
            self.text_template,
            self.instruction,
            choices,
            batch_size,
        )
        return torch.tensor(
            [
                [answer.probabilities[label] for label in label_set]
                for answer in answers
            ]
        )
