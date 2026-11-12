from pathlib import Path

import pytest
from transformers import PreTrainedTokenizerBase

from kiyal.tokenizer import REPLACEMENT_CHARACTER, IncrementalDetokenizer, Tokenizer

TEXTS = [
    "The capital of France is",
    "def fibonacci(n):\n    return n",
    "Кыял деген эмне? Көңүл, өмүр, таң.",
    "今天天气很好。",
]
MESSAGES = [{"role": "user", "content": "Hi"}]

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tokenizer(smollm2_checkpoint: Path) -> Tokenizer:
    return Tokenizer.from_dir(smollm2_checkpoint)


def test_smollm2_special_tokens(tokenizer: Tokenizer) -> None:
    assert tokenizer.vocab_size == 49152
    assert tokenizer.decode([1, 2], skip_special_tokens=False) == "<|im_start|><|im_end|>"
    assert (tokenizer.bos_token_id, tokenizer.eos_token_id) == (1, 2)


@pytest.mark.parametrize("text", TEXTS)
def test_smollm2_encoding_matches_transformers(
    tokenizer: Tokenizer, smollm2_tokenizer: PreTrainedTokenizerBase, text: str
) -> None:
    ids = tokenizer.encode(text)
    assert ids == smollm2_tokenizer(text)["input_ids"]
    assert tokenizer.decode(ids) == text


@pytest.mark.parametrize("text", TEXTS)
def test_smollm2_stream_adds_up_to_the_text(tokenizer: Tokenizer, text: str) -> None:
    detokenizer = IncrementalDetokenizer(tokenizer)
    pieces = [detokenizer.add(token_id) for token_id in tokenizer.encode(text)]
    assert "".join(pieces) == text
    assert detokenizer.finish() == ""
    assert not any(REPLACEMENT_CHARACTER in piece for piece in pieces)


def test_smollm2_chat_prompt(tokenizer: Tokenizer) -> None:
    # SmolLM2 uses ChatML and inserts a default system message when the chat has none.
    text = tokenizer.apply_chat_template(MESSAGES)
    assert text == (
        "<|im_start|>system\n"
        "You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    ids = tokenizer.encode_chat(MESSAGES)
    assert tokenizer.decode(ids, skip_special_tokens=False) == text
    assert ids.count(1) == 3
