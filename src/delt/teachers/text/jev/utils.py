from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from typesafe_sdk import ChoiceAnswer, TypeSafeClient


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
            "classification": {
                "type": "choice",
                "instructions": instruction,
                "criteria": choices,
            },
        },
    )
    return result.answers["classification"]


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
