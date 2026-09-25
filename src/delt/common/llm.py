"""Module for litellm utils."""

import json
from concurrent.futures import ThreadPoolExecutor

import litellm
from pydantic import BaseModel
from tqdm import tqdm


def generate_completion(
    model_name: str,
    conversation: list[dict],
    response_format: BaseModel,
    decoding_args: dict | None = None,
    extra_body: dict | None = None,
) -> BaseModel:
    """
    Generate a completion using the specified model for a conversation.

    Args:
        model_name (str): name of the model.
        conversation (list[dict]): message thread.
        response_format (BaseModel): pydantic model for response formatting.
        decoding_args (Optional[dict]): additional arguments for decoding.
        extra_body (Optional[dict]): extra body to send to the model.

    Returns:
        BaseModel: a completion output.

    """
    if decoding_args is None:
        decoding_args = {}

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
    decoding_args: dict | None = None,
    batch_size: int = 4,
    extra_body: list[dict] | None = None,
) -> list[BaseModel]:
    """
    Generate completions using the specified model and conversations.

    Args:
        model_name (str): name of the model.
        conversations (list[list[dict]]): a list of conversations.
        response_format (BaseModel): pydantic model for response formatting.
        decoding_args (Optional[dict]): additional arguments for decoding.
        batch_size (int): number of concurrent requests.
        extra_body (Optional[list[dict]]): extra bodies to send to the model.

    Returns:
        list[BaseModel]: completion outputs.

    """
    if decoding_args is None:
        decoding_args = {}

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
                    extra_body[i] if extra_body is not None else None,
                )
            )
        completions = [response.result() for response in tqdm(responses)]

    return completions
