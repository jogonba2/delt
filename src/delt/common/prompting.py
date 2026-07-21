"""Module for prompting utils."""

from typing import Literal

from pydantic import BaseModel, create_model


def format_prompt(
    label_verbalizations: dict[str, str], prompt_template: str
) -> list[str]:
    """
    Format a prompt template with label verbalizations.

    Args:
        label_verbalizations (dict[str, str]): verbalizations of the labels, e.g. {"positive": "very cool!", "negative": "horrible"}
        prompt_template (str): template to format label verbalizations, e.g., "This text is {}" being instantiated as "This text is very cool!".

    Returns:
        list[str]: instantiated prompts, one for each verbalization in `label_verbalizations`.

    """
    return [
        prompt_template.format(verbalization)
        for verbalization in label_verbalizations.values()
    ]


def make_output_model(label_set: list[str]) -> type[BaseModel]:
    """
    Create a pydantic model dynamically to be used as structured output in classification tasks.

    Args:
        label_set (list[str]): list of label names.

    Returns:
        type[BaseModel]: a pydantic model.

    """
    return create_model(
        "Output", label=(Literal.__getitem__(tuple(label_set)), ...)
    )
