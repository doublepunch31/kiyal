import asyncio
import contextlib
import time
import uuid
from collections import abc
from http import HTTPStatus
from typing import Final

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from starlette.types import Receive, Scope, Send

from kiyal import __version__
from kiyal.engine.async_engine import AsyncEngine, EngineStoppedError
from kiyal.engine.sequence import RequestOutput, SamplingParams
from kiyal.server.metrics import Metrics
from kiyal.server.protocol import (
    AssistantMessage,
    ChatChoice,
    ChatChunkChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatDelta,
    Completion,
    CompletionChoice,
    CompletionRequest,
    ErrorDetail,
    ErrorResponse,
    ModelCard,
    ModelList,
    SamplingRequest,
    Usage,
)
from kiyal.tokenizer import Tokenizer, TokenizerError

# OpenAI's default for /v1/completions.
DEFAULT_COMPLETION_TOKENS: Final = 16
# The status that servers such as nginx log for a client that left before the response; no
# client ever receives it.
CLIENT_CLOSED_REQUEST: Final = 499


class APIError(Exception):
    """An error that the API reports to the client in OpenAI's error format."""

    def __init__(
        self, status_code: int, message: str, *, param: str | None = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.param = param
        self.code = code

    def body(self) -> ErrorResponse:
        server_side = self.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR
        kind = "server_error" if server_side else "invalid_request_error"
        detail = ErrorDetail(message=self.message, type=kind, param=self.param, code=self.code)
        return ErrorResponse(error=detail)

    def response(self, headers: abc.Mapping[str, str] | None = None) -> Response:
        return _json(self.body(), self.status_code, headers)


def build_app(engine: AsyncEngine, tokenizer: Tokenizer, model_name: str) -> FastAPI:
    """Builds the HTTP API around an engine.

    The app starts the engine's step loop when it starts and stops it when it shuts down.

    Args:
        engine: The engine that generates the outputs, not started yet.
        tokenizer: Encodes prompts. Give the app its own instance rather than the one the
            engine decodes with: the app tokenizes on the event loop's thread while the
            engine decodes on its own, and one instance is not safe to share across
            threads for every tokenizer configuration.
        model_name: The name clients must give in the `model` field.
    """
    server = _Server(engine, tokenizer, model_name)
    app = FastAPI(title="Kiyal", version=__version__, lifespan=server.lifespan)
    app.add_api_route("/v1/completions", server.completions, methods=["POST"])
    app.add_api_route("/v1/chat/completions", server.chat_completions, methods=["POST"])
    app.add_api_route("/v1/models", server.models, methods=["GET"])
    app.add_api_route("/health", server.health, methods=["GET"])
    app.add_api_route("/metrics", server.metrics, methods=["GET"])
    app.add_exception_handler(APIError, _api_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(HTTPException, _http_error)
    app.add_exception_handler(Exception, _internal_error)
    return app


class _Server:
    def __init__(self, engine: AsyncEngine, tokenizer: Tokenizer, model_name: str) -> None:
        self._engine = engine
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._created = int(time.time())
        self._metrics = Metrics(engine.stats)

    @contextlib.asynccontextmanager
    async def lifespan(self, app: FastAPI) -> abc.AsyncGenerator[None, None]:
        self._engine.start()
        try:
            yield
        finally:
            self._engine.shutdown()

    async def health(self) -> Response:
        if not self._engine.is_running:
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, "the engine has stopped")
        return Response(status_code=HTTPStatus.OK)

    async def metrics(self) -> Response:
        return Response(self._metrics.render(), media_type=CONTENT_TYPE_LATEST)

    async def models(self) -> Response:
        card = ModelCard(id=self._model_name, created=self._created)
        return _json(ModelList(data=[card]))

    async def completions(self, body: CompletionRequest, request: Request) -> Response:
        return await _cancel_on_disconnect(request, self._completions(body))

    async def chat_completions(self, body: ChatCompletionRequest, request: Request) -> Response:
        return await _cancel_on_disconnect(request, self._chat_completions(body))

    async def _completions(self, body: CompletionRequest) -> Response:
        self._check(body)
        if body.echo:
            raise APIError(HTTPStatus.BAD_REQUEST, "echo is not supported", param="echo")
        if body.logprobs is not None:
            raise APIError(HTTPStatus.BAD_REQUEST, "logprobs are not supported", param="logprobs")
        max_tokens = DEFAULT_COMPLETION_TOKENS if body.max_tokens is None else body.max_tokens
        params = _sampling_params(body, max_tokens)
        prompt = body.prompt
        prompt_token_ids = self._tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        num_prompt_tokens = len(prompt_token_ids)
        request_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        outputs = await self._start(request_id, prompt_token_ids, params)

        def completion(choices: list[CompletionChoice], usage: Usage | None) -> Completion:
            return Completion(
                id=request_id, created=created, model=self._model_name, choices=choices, usage=usage
            )

        if body.stream:
            events = _events(
                outputs,
                lambda text, reason: completion(
                    [CompletionChoice(index=0, text=text, finish_reason=reason)], None
                ),
                lambda num_tokens: completion([], _usage(num_prompt_tokens, num_tokens)),
                include_usage=_include_usage(body),
            )
            return EventStreamResponse(events)
        text, num_tokens, reason = await _collect(outputs)
        choice = CompletionChoice(index=0, text=text, finish_reason=reason)
        return _json(completion([choice], _usage(num_prompt_tokens, num_tokens)))

    async def _chat_completions(self, body: ChatCompletionRequest) -> Response:
        self._check(body)
        if body.logprobs:
            raise APIError(HTTPStatus.BAD_REQUEST, "logprobs are not supported", param="logprobs")
        max_tokens = body.max_completion_tokens
        if max_tokens is None:
            max_tokens = body.max_tokens
        if max_tokens is None:
            # As in OpenAI's API, the reply may fill the rest of the context; the engine
            # lowers this to the room the prompt leaves.
            max_tokens = self._engine.max_model_len
        params = _sampling_params(body, max_tokens)
        messages = [{"role": message.role, "content": message.content} for message in body.messages]
        try:
            prompt_token_ids = self._tokenizer.encode_chat(messages)
        except TokenizerError as e:
            raise APIError(HTTPStatus.BAD_REQUEST, str(e), param="messages") from e
        num_prompt_tokens = len(prompt_token_ids)
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        outputs = await self._start(request_id, prompt_token_ids, params)

        def chunk(choices: list[ChatChunkChoice], usage: Usage | None) -> ChatCompletionChunk:
            return ChatCompletionChunk(
                id=request_id, created=created, model=self._model_name, choices=choices, usage=usage
            )

        if body.stream:
            events = _events(
                outputs,
                lambda text, reason: chunk(
                    [ChatChunkChoice(index=0, delta=ChatDelta(content=text), finish_reason=reason)],
                    None,
                ),
                lambda num_tokens: chunk([], _usage(num_prompt_tokens, num_tokens)),
                include_usage=_include_usage(body),
                # OpenAI's first chunk names the role and carries no text yet.
                opening=chunk(
                    [
                        ChatChunkChoice(
                            index=0,
                            delta=ChatDelta(role="assistant", content=""),
                            finish_reason=None,
                        )
                    ],
                    None,
                ),
            )
            return EventStreamResponse(events)
        text, num_tokens, reason = await _collect(outputs)
        choice = ChatChoice(index=0, message=AssistantMessage(content=text), finish_reason=reason)
        completion = ChatCompletion(
            id=request_id,
            created=created,
            model=self._model_name,
            choices=[choice],
            usage=_usage(num_prompt_tokens, num_tokens),
        )
        return _json(completion)

    def _check(self, body: SamplingRequest) -> None:
        if body.model != self._model_name:
            raise APIError(
                HTTPStatus.NOT_FOUND,
                f"the model {body.model!r} does not exist; this server serves {self._model_name!r}",
                param="model",
                code="model_not_found",
            )
        if body.n is not None and body.n != 1:
            raise APIError(HTTPStatus.BAD_REQUEST, "only n=1 is supported", param="n")
        if body.stream_options is not None and not body.stream:
            raise APIError(
                HTTPStatus.BAD_REQUEST,
                "stream_options is only allowed when stream is true",
                param="stream_options",
            )

    async def _start(
        self, request_id: str, prompt_token_ids: list[int], params: SamplingParams
    ) -> abc.AsyncGenerator[RequestOutput, None]:
        """Adds a request, waits for its first output, and returns all its outputs.

        Waiting for the first output turns a request that the engine rejects into an error
        response before any part of a streamed response has been sent.
        """
        arrival = time.perf_counter()
        outputs = self._engine.generate(request_id, prompt_token_ids, params)
        try:
            first = await anext(outputs)
        except ValueError as e:
            raise APIError(HTTPStatus.BAD_REQUEST, str(e)) from e
        except EngineStoppedError as e:
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, str(e)) from e
        self._metrics.time_to_first_token.observe(time.perf_counter() - arrival)
        self._metrics.prompt_tokens.inc(len(prompt_token_ids))
        return self._measured(first, outputs, arrival)

    async def _measured(
        self,
        first: RequestOutput,
        rest: abc.AsyncGenerator[RequestOutput, None],
        arrival: float,
    ) -> abc.AsyncGenerator[RequestOutput, None]:
        """Yields `first`, then the outputs of `rest`, and records their latencies.

        `rest` is closed however this ends, which aborts the request if it has not
        finished.
        """
        metrics = self._metrics
        last_time = time.perf_counter()
        finish_reason = "abort"
        try:
            output = first
            while True:
                metrics.generation_tokens.inc()
                if output.finish_reason is not None:
                    metrics.e2e_request_latency.observe(last_time - arrival)
                    finish_reason = output.finish_reason.value
                yield output
                try:
                    output = await anext(rest)
                except StopAsyncIteration:
                    break
                now = time.perf_counter()
                metrics.inter_token_latency.observe(now - last_time)
                last_time = now
        finally:
            metrics.requests_finished.labels(finish_reason=finish_reason).inc()
            await rest.aclose()


class EventStreamResponse(StreamingResponse):
    """Server-sent events from a generator that is closed when the response ends.

    Starlette stops iterating the body when the client disconnects but does not close it.
    If the disconnect comes while an event is being sent, the generator is left at a
    yield, and its cleanup, which aborts the request, would wait until the generator is
    garbage collected. Closing it here aborts the request at once.
    """

    def __init__(self, events: abc.AsyncGenerator[str, None]) -> None:
        super().__init__(events, media_type="text/event-stream")
        self._events = events

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._events.aclose()


async def _cancel_on_disconnect(request: Request, handler: abc.Awaitable[Response]) -> Response:
    """Runs a handler, and cancels it if the client disconnects before it returns.

    Starlette notices a disconnect only while it sends a streamed response. Before the
    handler returns, nothing reads from the connection, so a client that gave up on a
    response, or on a stream before its first token, would leave its request running to
    the end. Cancelling the handler aborts the request it waits for.
    """
    work = asyncio.ensure_future(handler)
    watcher = asyncio.ensure_future(_wait_for_disconnect(request))
    try:
        await asyncio.wait({work, watcher}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        # Both tasks end before this returns, also when this task is cancelled, as on
        # shutdown. The watcher must be gone before a streamed response starts reading
        # the connection itself.
        for task in (watcher, work):
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
    if work.cancelled():
        return Response(status_code=CLIENT_CLOSED_REQUEST)
    return work.result()


async def _wait_for_disconnect(request: Request) -> None:
    # FastAPI has read the whole body before the handler runs, so the next message is the
    # disconnect.
    while (await request.receive())["type"] != "http.disconnect":
        pass


async def _collect(outputs: abc.AsyncGenerator[RequestOutput, None]) -> tuple[str, int, str]:
    """Waits for a request's whole output.

    Returns:
        The text, the number of generated tokens and the finish reason.
    """
    texts: list[str] = []
    reason: str | None = None
    try:
        async for output in outputs:
            texts.append(output.text)
            if output.finish_reason is not None:
                reason = output.finish_reason.value
    except EngineStoppedError as e:
        raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, str(e)) from e
    assert reason is not None, "a request's last output says why it ended"
    return "".join(texts), len(texts), reason


async def _events(
    outputs: abc.AsyncGenerator[RequestOutput, None],
    chunk: abc.Callable[[str, str | None], BaseModel],
    usage_chunk: abc.Callable[[int], BaseModel],
    *,
    include_usage: bool,
    opening: BaseModel | None = None,
) -> abc.AsyncGenerator[str, None]:
    """Streams a request's outputs as server-sent events, ending with `data: [DONE]`.

    Args:
        outputs: The request's outputs.
        chunk: Builds an event from new text and, on the last output, the finish reason.
        usage_chunk: Builds the event with the token counts from the number of generated
            tokens.
        include_usage: Whether to send that event before the end.
        opening: An event to send before the first output.
    """
    if opening is not None:
        yield _event(opening)
    num_tokens = 0
    try:
        async for output in outputs:
            num_tokens += 1
            reason = None if output.finish_reason is None else output.finish_reason.value
            # A token whose text is held back adds nothing to send, unless it ends the output.
            if output.text or reason is not None:
                yield _event(chunk(output.text, reason))
    except EngineStoppedError as e:
        # The status line has been sent already, so the error goes into the stream.
        yield _event(APIError(HTTPStatus.SERVICE_UNAVAILABLE, str(e)).body())
    else:
        if include_usage:
            yield _event(usage_chunk(num_tokens))
    yield "data: [DONE]\n\n"


def _event(body: BaseModel) -> str:
    return f"data: {body.model_dump_json()}\n\n"


def _include_usage(body: SamplingRequest) -> bool:
    return body.stream_options is not None and bool(body.stream_options.include_usage)


def _sampling_params(body: SamplingRequest, max_tokens: int) -> SamplingParams:
    """Builds the engine's sampling settings, with OpenAI's defaults for fields set to null.

    Ranges are checked by SamplingParams, so the API accepts what the engine accepts.
    """
    stop = body.stop
    try:
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=1.0 if body.temperature is None else body.temperature,
            top_k=body.top_k,
            top_p=1.0 if body.top_p is None else body.top_p,
            min_p=body.min_p,
            repetition_penalty=body.repetition_penalty,
            presence_penalty=0.0 if body.presence_penalty is None else body.presence_penalty,
            frequency_penalty=0.0 if body.frequency_penalty is None else body.frequency_penalty,
            seed=body.seed,
            stop=() if stop is None else (stop,) if isinstance(stop, str) else tuple(stop),
            ignore_eos=body.ignore_eos,
        )
    except ValueError as e:
        raise APIError(HTTPStatus.BAD_REQUEST, str(e)) from e


def _usage(num_prompt_tokens: int, num_output_tokens: int) -> Usage:
    # Every generated token counts, including a stop token, which adds no text.
    return Usage(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_output_tokens,
        total_tokens=num_prompt_tokens + num_output_tokens,
    )


def _json(
    body: BaseModel, status_code: int = HTTPStatus.OK, headers: abc.Mapping[str, str] | None = None
) -> Response:
    return Response(
        body.model_dump_json(),
        status_code=status_code,
        headers=headers,
        media_type="application/json",
    )


async def _api_error(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, APIError)
    return exc.response()


async def _validation_error(request: Request, exc: Exception) -> Response:
    # FastAPI answers 422; OpenAI's API answers 400 for a malformed request.
    assert isinstance(exc, RequestValidationError)
    messages: list[str] = []
    fields: list[str] = []
    for error in exc.errors():
        # The location starts with where the value came from, such as "body", and goes on
        # with the field, then parts inside it; for a union, the type that was tried.
        path = [str(part) for part in error["loc"][1:]]
        messages.append(f"{'.'.join(path)}: {error['msg']}" if path else str(error["msg"]))
        fields.append(path[0] if path else "")
    return APIError(HTTPStatus.BAD_REQUEST, "; ".join(messages), param=fields[0] or None).response()


async def _http_error(request: Request, exc: Exception) -> Response:
    # Unknown paths and methods.
    assert isinstance(exc, HTTPException)
    return APIError(exc.status_code, str(exc.detail)).response(exc.headers)


async def _internal_error(request: Request, exc: Exception) -> Response:
    # The server still logs the exception with its traceback.
    return APIError(HTTPStatus.INTERNAL_SERVER_ERROR, "internal server error").response()
