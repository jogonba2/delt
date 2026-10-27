"""Module for prompting utils."""

from typing import Literal

from pydantic import BaseModel, Field, create_model


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


def make_output_model(
    label_set: list[str],
    with_reasoning: bool = False,
) -> type[BaseModel]:
    """
    Create a pydantic model dynamically to be used as structured output in classification tasks.

    Args:
        label_set (list[str]): list of label names.
        with_reasoning (bool): if True, add a `reasoning` field generated before the label.

    Returns:
        type[BaseModel]: a pydantic model.

    """
    LabelType = Literal[tuple(label_set)]
    fields = {}
    if with_reasoning:
        fields["reasoning"] = (
            str,
            Field(
                ...,
                description="Brief justification for the chosen label, written before deciding.",
            ),
        )
    fields["label"] = (
        LabelType,
        Field(
            ..., description="The single label that best describes the input."
        ),
    )
    return create_model("Output", **fields)
