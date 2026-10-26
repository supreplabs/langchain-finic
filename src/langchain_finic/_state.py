"""Normalize LangChain messages without sending framework bookkeeping to Finic."""

import math
from collections.abc import Mapping, Sequence

from langchain_core.messages import BaseMessage, convert_to_openai_messages


def normalize_state(value):
    if isinstance(value, BaseMessage):
        return normalize_state(convert_to_openai_messages(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("State object keys must be strings")
        return {key: normalize_state(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [normalize_state(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("State must contain JSON values or LangChain messages")
