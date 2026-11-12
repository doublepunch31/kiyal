from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, Protocol, Self, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase

TOKENIZER_FILE: Final = "tokenizer.json"
# What a decoder writes for bytes that are not a complete UTF-8 character.
REPLACEMENT_CHARACTER: Final = "\ufffd"
# Prompt tokens decoded along with the first output tokens. One visible token is enough
# context; a few more make up for special tokens, which decode to nothing.
PROMPT_CONTEXT_TOKENS: Final = 8


class TokenizerError(Exception):
    """Raised when tokenizer files are missing or a model has no chat template."""


class _Backend(Protocol):
    # The part of transformers' tokenizer interface used here. Its methods take unannotated
    # keyword arguments, which strict type checking reports on every call.
    bos_token_id: int | None
    eos_token_id: int | None
    chat_template: object

    def __len__(self) -> int: ...
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str: ...
    def apply_chat_template(
        self, conversation: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str: ...


class Tokenizer:
    """Converts between text and token ids with a model's Hugging Face tokenizer."""

    def __init__(self, backend: PreTrainedTokenizerBase) -> None:
        self._backend = cast(_Backend, backend)

    @classmethod
    def from_dir(cls, model_dir: Path) -> Self:
        """Loads the tokenizer stored in a local checkpoint directory.

        Raises:
            TokenizerError: If the directory has no tokenizer.json.
        """
        if not (model_dir / TOKENIZER_FILE).is_file():
            raise TokenizerError(f"no {TOKENIZER_FILE} in {model_dir}")
        # from_pretrained is unannotated, so its result is cast to the class it returns.
        load = cast(Callable[[Path], PreTrainedTokenizerBase], AutoTokenizer.from_pretrained)  # pyright: ignore[reportUnknownMemberType]
        return cls(load(model_dir))

    @property
    def vocab_size(self) -> int:
        """Number of token ids, counting special tokens added to the base vocabulary."""
        return len(self._backend)

    @property
    def bos_token_id(self) -> int | None:
        return self._backend.bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self._backend.eos_token_id

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Tokenizes text.

        Args:
            text: The text to tokenize.
            add_special_tokens: Whether to add the tokens the model expects around plain text,
                such as a leading BOS. Pass False for text that already contains them.
        """
        return self._backend.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Converts token ids to text.

        Bytes that do not form a complete UTF-8 character decode to U+FFFD.
        """
        return self._backend.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool = True
    ) -> str:
        """Renders chat messages into the prompt format the model was trained on.

        Args:
            messages: Messages with "role" and "content" keys, oldest first.
            add_generation_prompt: Whether to end with the header that opens the assistant's
                reply, so that the model answers instead of continuing the last message.

        Raises:
            TokenizerError: If the model has no chat template.
        """
        if self._backend.chat_template is None:
            raise TokenizerError("the tokenizer has no chat template")
        return self._backend.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def encode_chat(
        self, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool = True
    ) -> list[int]:
        """Renders chat messages and tokenizes the result.

        Raises:
            TokenizerError: If the model has no chat template.
        """
        text = self.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
        # The template writes its own special tokens; adding them again would duplicate BOS.
        return self.encode(text, add_special_tokens=False)


class StopStringMatcher:
    """Finds stop strings in streamed text.

    Text arrives in pieces, and a stop string can start in one piece and end in a later
    one. Text that could still turn into a stop string is held back until the following
    pieces decide it.
    """

    def __init__(self, stop: Sequence[str]) -> None:
        """Starts a stream.

        Raises:
            ValueError: If a stop string is empty.
        """
        if any(not stop_string for stop_string in stop):
            raise ValueError("stop strings must not be empty")
        self._stop = tuple(stop)
        self._held = ""
        self._stopped = False

    @property
    def stopped(self) -> bool:
        """Whether a stop string has been found. Text added after that is dropped."""
        return self._stopped

    def add(self, text: str) -> str:
        """Adds a piece of text and returns the text that is safe to show.

        The returned text ends before the first stop string and never holds part of one.
        """
        if self._stopped:
            return ""
        # A stop string that ends in this piece cannot start before the held text: its
        # earlier characters would have been the start of a stop string and been held.
        buffer = self._held + text
        found = [index for index in map(buffer.find, self._stop) if index >= 0]
        if found:
            self._stopped = True
            self._held = ""
            return buffer[: min(found)]
        safe_length = len(buffer) - self._possible_stop_length(buffer)
        self._held = buffer[safe_length:]
        return buffer[:safe_length]

    def finish(self) -> str:
        """Returns the text still held back, for when generation ends without a stop."""
        held, self._held = self._held, ""
        return held

    def _possible_stop_length(self, text: str) -> int:
        # Length of the longest ending of the text that is the beginning of a stop string.
        longest = 0
        for stop_string in self._stop:
            for length in range(min(len(stop_string) - 1, len(text)), longest, -1):
                if text.endswith(stop_string[:length]):
                    longest = length
                    break
        return longest


class IncrementalDetokenizer:
    """Turns generated tokens into text one token at a time.

    A token cannot be decoded on its own. The UTF-8 bytes of one character can be spread
    over several tokens, and some tokenizers write a word's leading space only when the token
    is not the first one decoded. So each token is decoded together with the tokens before
    it, and only the text it adds is returned.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        prompt_token_ids: Sequence[int] = (),
        stop: Sequence[str] = (),
    ) -> None:
        """Starts a stream.

        Args:
            tokenizer: The tokenizer that produced the token ids.
            prompt_token_ids: The prompt the output continues. Its last tokens are the
                context for decoding the first output tokens; their text is not returned.
            stop: Strings that end the output. The output stops before the first one found
                and does not include it.

        Raises:
            ValueError: If a stop string is empty.
        """
        self._tokenizer = tokenizer
        self._stop = StopStringMatcher(stop)
        # Tokens whose text has already been returned, kept as context for the next decode.
        self._context = list(prompt_token_ids[-PROMPT_CONTEXT_TOKENS:])
        # Tokens whose text ends inside a character and is held back until it is complete.
        self._pending: list[int] = []

    @property
    def stopped(self) -> bool:
        """Whether a stop string has been found. Tokens added after that give no text."""
        return self._stop.stopped

    def add(self, token_id: int) -> str:
        """Adds a generated token and returns the text that became complete.

        Returns an empty string while the output ends inside a character or in what could
        be the start of a stop string.
        """
        self._pending.append(token_id)
        text = self._pending_text()
        if text.endswith(REPLACEMENT_CHARACTER):
            return ""
        self._context, self._pending = self._pending, []
        return self._stop.add(text)

    def finish(self) -> str:
        """Returns the text still held back, for when generation ends.

        A character that never got its remaining bytes comes out as U+FFFD.
        """
        text = self._pending_text()
        self._context, self._pending = self._context + self._pending, []
        return self._stop.add(text) + self._stop.finish()

    def _pending_text(self) -> str:
        context_text = self._tokenizer.decode(self._context)
        text = self._tokenizer.decode(self._context + self._pending)
        return text[len(context_text) :]
