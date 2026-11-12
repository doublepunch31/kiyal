import shutil
from pathlib import Path

import pytest
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from kiyal.tokenizer import (
    REPLACEMENT_CHARACTER,
    IncrementalDetokenizer,
    Tokenizer,
    TokenizerError,
)

SAMPLES = {
    "english": "The quick brown fox jumps over the lazy dog , twice !",
    "russian": "Съешь же ещё этих мягких французских булок, да выпей чаю.",
    "kyrgyz": "Кыргыз тилинде ң, ө жана ү тамгалары бар: көңүл, өмүр, жүрөк, таң.",
    "chinese": "今天天气很好，我们去公园散步。",
    # A rocket, a thumbs-up with a skin-tone modifier, and a woman joined to a laptop by a
    # zero-width joiner: one, two and three code points per glyph, four UTF-8 bytes each.
    "emoji": "go \U0001f680 ok \U0001f44d\U0001f3fd dev \U0001f469‍\U0001f4bb",
    "whitespace": "  two spaces,\ttab\nnewline  ",
}
# Scripts whose characters take several UTF-8 bytes, which the tiny tokenizer mostly encodes
# as one token per byte.
MULTIBYTE_SAMPLES = ["russian", "kyrgyz", "chinese", "emoji"]
ROCKET = "\U0001f680"
MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Салам!"},
]


@pytest.mark.parametrize("text", SAMPLES.values(), ids=list(SAMPLES))
def test_decoding_restores_the_encoded_text(tiny_tokenizer: Tokenizer, text: str) -> None:
    assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text


def test_special_tokens_are_part_of_the_vocabulary(tiny_tokenizer: Tokenizer) -> None:
    bos, eos = tiny_tokenizer.bos_token_id, tiny_tokenizer.eos_token_id
    assert bos is not None
    assert eos is not None
    assert bos != eos
    assert max(bos, eos) < tiny_tokenizer.vocab_size
    assert max(tiny_tokenizer.encode(SAMPLES["kyrgyz"])) < tiny_tokenizer.vocab_size


def test_encode_adds_bos_unless_disabled(tiny_tokenizer: Tokenizer) -> None:
    with_special = tiny_tokenizer.encode("the cat")
    without_special = tiny_tokenizer.encode("the cat", add_special_tokens=False)
    assert with_special == [tiny_tokenizer.bos_token_id, *without_special]


def test_decode_skips_special_tokens_unless_asked(tiny_tokenizer: Tokenizer) -> None:
    ids = tiny_tokenizer.encode("the cat")
    assert tiny_tokenizer.decode(ids) == "the cat"
    assert tiny_tokenizer.decode(ids, skip_special_tokens=False) == "<|bos|>the cat"


def test_chat_template_renders_each_message(tiny_tokenizer: Tokenizer) -> None:
    assert tiny_tokenizer.apply_chat_template(MESSAGES) == (
        "<|bos|>system\nBe brief.<|eos|>\n<|bos|>user\nСалам!<|eos|>\n<|bos|>assistant\n"
    )


def test_chat_template_can_omit_the_generation_prompt(tiny_tokenizer: Tokenizer) -> None:
    text = tiny_tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=False)
    assert text.endswith("Салам!<|eos|>\n")


def test_encode_chat_does_not_add_a_second_bos(tiny_tokenizer: Tokenizer) -> None:
    ids = tiny_tokenizer.encode_chat(MESSAGES)
    text = tiny_tokenizer.apply_chat_template(MESSAGES)
    bos = tiny_tokenizer.bos_token_id
    assert bos is not None
    assert ids == tiny_tokenizer.encode(text, add_special_tokens=False)
    # One BOS per message and one for the generation prompt, all written by the template.
    assert ids.count(bos) == len(MESSAGES) + 1
    assert tiny_tokenizer.decode(ids, skip_special_tokens=False) == text


def test_missing_chat_template_raises(tiny_tokenizer_dir: Path, tmp_path: Path) -> None:
    model_dir = tmp_path / "no-template"
    shutil.copytree(tiny_tokenizer_dir, model_dir)
    (model_dir / "chat_template.jinja").unlink()
    tokenizer = Tokenizer.from_dir(model_dir)
    assert not tokenizer.has_chat_template
    assert Tokenizer.from_dir(tiny_tokenizer_dir).has_chat_template
    with pytest.raises(TokenizerError, match="no chat template"):
        tokenizer.apply_chat_template(MESSAGES)
    with pytest.raises(TokenizerError, match="no chat template"):
        tokenizer.encode_chat(MESSAGES)


def test_directory_without_tokenizer_raises(tmp_path: Path) -> None:
    with pytest.raises(TokenizerError, match=r"no tokenizer\.json"):
        Tokenizer.from_dir(tmp_path)


def stream(
    tokenizer: Tokenizer, token_ids: list[int], prompt: list[int] | None = None
) -> list[str]:
    """Feeds tokens one at a time and returns the text of each step, then of finish()."""
    detokenizer = IncrementalDetokenizer(tokenizer, prompt or [])
    return [*(detokenizer.add(token_id) for token_id in token_ids), detokenizer.finish()]


@pytest.mark.parametrize("name", MULTIBYTE_SAMPLES)
def test_single_tokens_can_hold_part_of_a_character(tiny_tokenizer: Tokenizer, name: str) -> None:
    # The reason for IncrementalDetokenizer: decoding these tokens one by one gives U+FFFD.
    ids = tiny_tokenizer.encode(SAMPLES[name], add_special_tokens=False)
    assert len(ids) > len(SAMPLES[name])
    assert any(REPLACEMENT_CHARACTER in tiny_tokenizer.decode([token_id]) for token_id in ids)


@pytest.mark.parametrize("text", SAMPLES.values(), ids=list(SAMPLES))
def test_stream_adds_up_to_the_decoded_text(tiny_tokenizer: Tokenizer, text: str) -> None:
    pieces = stream(tiny_tokenizer, tiny_tokenizer.encode(text, add_special_tokens=False))
    assert "".join(pieces) == text
    assert pieces[-1] == ""
    assert not any(REPLACEMENT_CHARACTER in piece for piece in pieces)


def test_a_character_is_held_back_until_its_last_byte(tiny_tokenizer: Tokenizer) -> None:
    ids = tiny_tokenizer.encode(f"a{ROCKET}b", add_special_tokens=False)
    # The rocket's four UTF-8 bytes are one token each.
    assert len(ids) == 6
    assert stream(tiny_tokenizer, ids) == ["a", "", "", "", ROCKET, "b", ""]


def test_finish_returns_an_unfinished_character_as_replacement(tiny_tokenizer: Tokenizer) -> None:
    ids = tiny_tokenizer.encode(f"a{ROCKET}", add_special_tokens=False)
    *pieces, flushed = stream(tiny_tokenizer, ids[:3])
    assert pieces == ["a", "", ""]
    assert flushed
    assert set(flushed) == {REPLACEMENT_CHARACTER}


def test_a_real_replacement_character_is_delayed_but_not_lost(tiny_tokenizer: Tokenizer) -> None:
    # U+FFFD in the text itself looks like an unfinished character until more text follows
    # or the stream ends.
    text = f"a{REPLACEMENT_CHARACTER}b{REPLACEMENT_CHARACTER}"
    pieces = stream(tiny_tokenizer, tiny_tokenizer.encode(text, add_special_tokens=False))
    assert "".join(pieces) == text
    assert pieces[-1] == REPLACEMENT_CHARACTER


def test_special_tokens_add_no_text(tiny_tokenizer: Tokenizer) -> None:
    eos = tiny_tokenizer.eos_token_id
    assert eos is not None
    ids = tiny_tokenizer.encode("the cat", add_special_tokens=False)
    pieces = stream(tiny_tokenizer, [*ids, eos, *ids])
    assert "".join(pieces) == "the catthe cat"
    assert pieces[len(ids)] == ""


def test_prompt_text_is_not_repeated(tiny_tokenizer: Tokenizer) -> None:
    prompt = tiny_tokenizer.encode(SAMPLES["kyrgyz"])
    output = tiny_tokenizer.encode(SAMPLES["chinese"], add_special_tokens=False)
    assert "".join(stream(tiny_tokenizer, output, prompt)) == SAMPLES["chinese"]


@pytest.fixture(scope="module")
def word_tokenizer() -> Tokenizer:
    """A SentencePiece-style tokenizer: tokens carry their leading space as a marker."""
    words = ["<unk>", "\u2581The", "\u2581capital", "\u2581is", "\u2581Paris"]
    vocab = {word: token_id for token_id, word in enumerate(words)}
    backend = BackendTokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Metaspace()
    backend.decoder = decoders.Metaspace()
    return Tokenizer(PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>"))


def test_earlier_output_is_the_context_for_later_tokens(word_tokenizer: Tokenizer) -> None:
    ids = word_tokenizer.encode("The capital is Paris")
    # Only the very first token of a text is written without its leading space.
    assert [word_tokenizer.decode([token_id]) for token_id in ids][1:] == ["capital", "is", "Paris"]
    assert stream(word_tokenizer, ids) == ["The", " capital", " is", " Paris", ""]


def test_prompt_context_keeps_the_first_leading_space(word_tokenizer: Tokenizer) -> None:
    prompt = word_tokenizer.encode("The capital is")
    output = word_tokenizer.encode("Paris")
    # Decoded on its own, the first output token loses the space that separates it from
    # the prompt.
    assert word_tokenizer.decode(output) == "Paris"
    assert stream(word_tokenizer, output) == ["Paris", ""]
    assert stream(word_tokenizer, output, prompt) == [" Paris", ""]
