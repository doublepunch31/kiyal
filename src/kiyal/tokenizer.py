from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, Protocol, Self, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase

TOKENIZER_FILE: Final = "tokenizer.json"


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
