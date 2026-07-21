"""Module for io utils."""

from base64 import b64encode
from typing import Optional


def base64_encode(data: bytes, mime_type: Optional[str] = None) -> str:
    """
    Encode a `bytes` object into base64.

    If `mime_type` is provided, a Data URI is returned with the mime type in the header.

    Args:
        data (bytes): bytes to encode.
        mime_type (Optional[str]): mime type of the data.

    Returns:
        str: a base64 stream in Data URI format if `mime_type` is provided.

    """
    b64 = b64encode(data).decode("utf-8")
    if mime_type:
        return f"data:{mime_type};base64,{b64}"
    return b64
