"""Large Language Model teacher for span classification."""

from delt.teachers.text.span.base import SpanTeacher
from delt.teachers.text.span.utils import make_spans_output_model
from delt.teachers.utils import generate_completions


class LLMSpanTextTeacher(SpanTeacher):
    """
    LLM-based teacher for span classification.

    Generates structured predictions from text inputs using LiteLLM.
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

    def predict(
        self, input_data: list[str], label_set: list[str], batch_size: int = 8
    ) -> list[list[tuple[str, str]]]:
        """
        Assign labels from `label_set` to the spans in `input_data` by using a teacher model.

        Args:
            input_data (list[T]): list of input data.
            label_set (list[str]): set of labels in your task.
            batch_size (int): batch size.

        Returns:
            list[list[tuple[str, str]]]: a list of spans for each input, e.g., [[("person", "Jose)]]

        """
        response_format = make_spans_output_model(label_set)
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
        return [
            [(span.text, span.label) for span in output.output]
            for output in responses
        ]
