from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_usage: bool | None = None


class SamplingRequest(BaseModel):
    """Fields that the completion endpoints share: the model and the sampling settings.

    Fields that OpenAI's API marks as nullable take None as their default value. Unknown
    fields are rejected rather than ignored, so a client does not believe it got a setting
    that was dropped.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool | None = None
    stream_options: StreamOptions | None = None
    n: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    # Not in OpenAI's API: the engine's other sampling settings.
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    ignore_eos: bool = False


class CompletionRequest(SamplingRequest):
    """Body of `POST /v1/completions`. The prompt is text or token ids."""

    prompt: str | list[int]
    max_tokens: int | None = None
    echo: bool | None = None
    logprobs: int | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(SamplingRequest):
    """Body of `POST /v1/chat/completions`."""

    messages: list[ChatMessage] = Field(min_length=1)
    # OpenAI's newer name for max_tokens; it wins when both are given.
    max_completion_tokens: int | None = None
    max_tokens: int | None = None
    logprobs: bool | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    index: int
    text: str
    logprobs: None = None
    finish_reason: str | None


class Completion(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage | None


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int
    message: AssistantMessage
    logprobs: None = None
    finish_reason: str


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


class ChatDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatChunkChoice(BaseModel):
    index: int
    delta: ChatDelta
    logprobs: None = None
    finish_reason: str | None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatChunkChoice]
    usage: Usage | None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "kiyal"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
