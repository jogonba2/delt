from typing import Literal

from pydantic import BaseModel, Field, create_model


def make_span_output_model(label_set: list[str]) -> type[BaseModel]:
    """
    Create a pydantic model dynamically for a single span.

    Args:
        label_set (list[str]): list of label names.

    Returns:
        type[BaseModel]: a pydantic model.

    """
    LabelType = Literal[tuple(label_set)]
    return create_model(
        "Span",
        text=(
            str,
            Field(
                ...,
                description="Exact text of the span as it appears in the input.",
            ),
        ),
        label=(
            LabelType,
            Field(..., description="Label assigned to the span."),
        ),
    )


def make_spans_output_model(label_set: list[str]) -> type[BaseModel]:
    """
    Create a pydantic model dynamically to be used as structured output in span classification

    Args:
        label_set (list[str]): list of label names.

    Returns:
        type[BaseModel]: a pydantic model.

    """
    span_model = make_span_output_model(label_set)
    return create_model("Output", output=(list[span_model], ...))
