import time
from typing import List, Optional, Union, Literal, Dict

from pydantic import BaseModel, Field


# **** For Request ****
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "observation"]
    content: str
    metadata: Optional[str] = None
    tools: Optional[List[dict]] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="model name")
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 409
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0

    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False

    # Addition parameters for chat/stream
    chunk: Optional[bool] = True

    # Additional parameters support for stop generation
    stop_token_ids: Optional[List[int]] = None
    repetition_penalty: Optional[float] = 1.1

    # Additional parameters supported by tools
    return_function_call: Optional[bool] = False

    # addition parameters for openai
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    logit_bias: Optional[Dict] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    n: Optional[int] = None


# **** For Response ****
class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: DeltaMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None
    logprobs: Optional[Dict] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionResponse(BaseModel):
    id: str
    choices: List[Union[ChatCompletionResponseStreamChoice, ChatCompletionResponseChoice]]
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field(..., description="model name")
    system_fingerprint: Optional[str] = None
    object: str = Union['chat.completion.chunk', 'chat.completion']
    usage: Optional[UsageInfo] = None

