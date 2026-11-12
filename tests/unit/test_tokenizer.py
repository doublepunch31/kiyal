import shutil
from pathlib import Path

import pytest

from kiyal.tokenizer import Tokenizer, TokenizerError

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
    with pytest.raises(TokenizerError, match="no chat template"):
        tokenizer.apply_chat_template(MESSAGES)
    with pytest.raises(TokenizerError, match="no chat template"):
        tokenizer.encode_chat(MESSAGES)


def test_directory_without_tokenizer_raises(tmp_path: Path) -> None:
    with pytest.raises(TokenizerError, match=r"no tokenizer\.json"):
        Tokenizer.from_dir(tmp_path)
