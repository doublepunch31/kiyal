import contextlib
import time
import uuid
from collections import abc
from http import HTTPStatus
from typing import Final

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from starlette.exceptions import HTTPException

from kiyal import __version__
from kiyal.engine.async_engine import AsyncEngine, EngineStoppedError
from kiyal.engine.sequence import RequestOutput, SamplingParams
from kiyal.server.protocol import (
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
from kiyal.tokenizer import Tokenizer

# OpenAI's default for /v1/completions.
DEFAULT_COMPLETION_TOKENS: Final = 16


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

    def response(self, headers: abc.Mapping[str, str] | None = None) -> Response:
        server_side = self.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR
        kind = "server_error" if server_side else "invalid_request_error"
        detail = ErrorDetail(message=self.message, type=kind, param=self.param, code=self.code)
        return _json(ErrorResponse(error=detail), self.status_code, headers)


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
    app.add_api_route("/v1/models", server.models, methods=["GET"])
    app.add_api_route("/health", server.health, methods=["GET"])
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

    async def models(self) -> Response:
        card = ModelCard(id=self._model_name, created=self._created)
        return _json(ModelList(data=[card]))

    async def completions(self, body: CompletionRequest) -> Response:
        self._check(body)
        if body.echo:
            raise APIError(HTTPStatus.BAD_REQUEST, "echo is not supported", param="echo")
        if body.logprobs is not None:
            raise APIError(HTTPStatus.BAD_REQUEST, "logprobs are not supported", param="logprobs")
        max_tokens = DEFAULT_COMPLETION_TOKENS if body.max_tokens is None else body.max_tokens
        params = _sampling_params(body, max_tokens)
        prompt = body.prompt
        prompt_token_ids = self._tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        request_id = f"cmpl-{uuid.uuid4().hex}"
        first, outputs = await self._start(request_id, prompt_token_ids, params)
        texts = [first.text]
        last = first
        try:
            async for output in outputs:
                texts.append(output.text)
                last = output
        except EngineStoppedError as e:
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, str(e)) from e
        assert last.finish_reason is not None, "a request's last output says why it ended"
        choice = CompletionChoice(
            index=0, text="".join(texts), finish_reason=last.finish_reason.value
        )
        usage = _usage(len(prompt_token_ids), len(texts))
        completion = Completion(
            id=request_id,
            created=int(time.time()),
            model=self._model_name,
            choices=[choice],
            usage=usage,
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

    async def _start(
        self, request_id: str, prompt_token_ids: list[int], params: SamplingParams
    ) -> tuple[RequestOutput, abc.AsyncGenerator[RequestOutput, None]]:
        """Adds a request and waits for its first output.

        Waiting for it turns a request that the engine rejects into an error response
        before any part of a streamed response has been sent.
        """
        outputs = self._engine.generate(request_id, prompt_token_ids, params)
        try:
            first = await anext(outputs)
        except ValueError as e:
            raise APIError(HTTPStatus.BAD_REQUEST, str(e)) from e
        except EngineStoppedError as e:
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, str(e)) from e
        return first, outputs


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
