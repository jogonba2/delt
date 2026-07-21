"""Module for litellm utils."""

import json
from concurrent.futures import ThreadPoolExecutor

import litellm
from pydantic import BaseModel
from tqdm import tqdm

from .logging import get_logger

_logger = get_logger(__name__)


def generate_completion(
    model_name: str,
    conversation: list[dict],
    response_format: BaseModel,
    decoding_args: dict = {},
    extra_body: dict = {},
) -> BaseModel:
    """
    Generate a completion using the specified model for a conversation.

    Args:
        model_name (str): name of the model.
        conversation (list[dict]): message thread.
        response_format (BaseModel): pydantic model for response formatting.
        decoding_args (dict): additional arguments for decoding.
        extra_body (dict): extra body to send to the model.

    Returns:
        BaseModel: a completion output.

    """
    response = litellm.completion(
        model=model_name,
        messages=conversation,
        response_format=response_format,
        extra_body=extra_body,
        **decoding_args,
    )

    if response.choices[0].finish_reason != "stop":
        raise RuntimeError(
            f"Completion did not finish properly: "
            f"{response.choices[0].finish_reason}"
        )

    json_response = json.loads(response.choices[0].message.content)
    return response_format(**json_response)


def generate_completions(
    model_name: str,
    conversations: list[list[dict]],
    response_format: BaseModel,
    decoding_args: dict = {},
    batch_size: int = 4,
    extra_body: list[dict] = [],
) -> list[BaseModel]:
    """
    Generate completions using the specified model and conversations.

    Args:
        model_name (str): name of the model.
        conversations (list[list[dict]]): a list of conversations.
        response_format (BaseModel): pydantic model for response formatting.
        decoding_args (dict): additional arguments for decoding.
        batch_size (int): number of concurrent requests.
        extra_body (list[dict]): extra bodies to send to the model.

    Returns:
        list[BaseModel]: completion outputs.

    """
    completions, responses = [], []
    with ThreadPoolExecutor(
        max_workers=min(batch_size, len(conversations))
    ) as thread_pool:
        for i, message in enumerate(conversations):
            responses.append(
                thread_pool.submit(
                    generate_completion,
                    model_name,
                    message,
                    response_format,
                    decoding_args,
                    extra_body[i] if extra_body else None,
                )
            )
        completions = [response.result() for response in tqdm(responses)]

    return completions
