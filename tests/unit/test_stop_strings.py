import random

import pytest

from kiyal.tokenizer import IncrementalDetokenizer, StopStringMatcher, Tokenizer


def feed(matcher: StopStringMatcher, pieces: list[str]) -> list[str]:
    """Adds the pieces in order and returns what each call gave back, then finish()."""
    return [*(matcher.add(piece) for piece in pieces), matcher.finish()]


def expected_output(text: str, stop: list[str]) -> str:
    found = [text.index(stop_string) for stop_string in stop if stop_string in text]
    return text[: min(found)] if found else text


def test_text_passes_through_without_stop_strings() -> None:
    matcher = StopStringMatcher([])
    assert feed(matcher, ["Hello", ", ", "world"]) == ["Hello", ", ", "world", ""]
    assert not matcher.stopped


def test_output_ends_before_a_stop_string_inside_one_piece() -> None:
    matcher = StopStringMatcher(["\n\n"])
    assert feed(matcher, ["first", " line\n\nsecond", " line"]) == ["first", " line", "", ""]
    assert matcher.stopped


def test_a_stop_string_split_over_pieces_is_never_shown() -> None:
    matcher = StopStringMatcher(["</end>"])
    outputs = feed(matcher, ["done ", "<", "/en", "d", ">", " more"])
    assert outputs == ["done ", "", "", "", "", "", ""]
    assert matcher.stopped


def test_a_false_start_is_released_once_it_cannot_match() -> None:
    matcher = StopStringMatcher(["</end>"])
    assert feed(matcher, ["a <", "/e", "m>", " b"]) == ["a ", "", "</em>", " b", ""]
    assert not matcher.stopped


def test_only_the_possible_start_is_held_back() -> None:
    # After "aaa" the output could still become "a" + "aab", so "aa" is held and "a" is shown.
    matcher = StopStringMatcher(["aab"])
    assert feed(matcher, ["a", "a", "a", "b", "c"]) == ["", "", "a", "", "", ""]
    assert matcher.stopped


def test_finish_releases_held_text() -> None:
    matcher = StopStringMatcher(["STOP"])
    assert feed(matcher, ["go ST"]) == ["go ", "ST"]
    assert not matcher.stopped


def test_the_earliest_stop_string_wins() -> None:
    matcher = StopStringMatcher(["later", "early"])
    assert feed(matcher, ["an early and a later one"]) == ["an ", ""]


def test_text_after_a_stop_is_dropped() -> None:
    matcher = StopStringMatcher(["."])
    assert matcher.add("One. Two") == "One"
    assert matcher.add(". Three") == ""
    assert matcher.finish() == ""


def test_empty_stop_string_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        StopStringMatcher(["ok", ""])


@pytest.mark.parametrize(
    ("text", "stop"),
    [
        ("The answer is 42.\nQuestion: next", ["\nQuestion:", "\n\n"]),
        ("көңүл, өмүр, жүрөк, таң", ["өмүр"]),
        ("今天天气很好，我们去公园散步。", ["我们", "，"]),
        ("abababac", ["abac", "bb"]),
        ("no stop string in here", ["absent", "here!"]),
        ("ends with a stop", ["stop"]),
    ],
)
def test_any_split_of_the_text_gives_the_same_output(text: str, stop: list[str]) -> None:
    rng = random.Random(0)
    for _ in range(200):
        cuts = sorted(rng.sample(range(len(text) + 1), k=rng.randint(0, min(6, len(text)))))
        pieces = [
            text[start:end] for start, end in zip([0, *cuts], [*cuts, len(text)], strict=True)
        ]
        matcher = StopStringMatcher(stop)
        outputs = feed(matcher, pieces)
        assert "".join(outputs) == expected_output(text, stop), pieces
        assert matcher.stopped == any(stop_string in text for stop_string in stop)
        # Nothing shown before the end may have to be taken back.
        shown = ""
        for output in outputs:
            shown += output
            assert expected_output(text, stop).startswith(shown)


def stream(tokenizer: Tokenizer, text: str, stop: list[str]) -> tuple[list[str], bool]:
    """Detokenizes the text's tokens one at a time, as generation would."""
    detokenizer = IncrementalDetokenizer(tokenizer, stop=stop)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    outputs = [*(detokenizer.add(token_id) for token_id in token_ids), detokenizer.finish()]
    return outputs, detokenizer.stopped


@pytest.mark.parametrize(
    ("text", "stop"),
    [
        ("the cat sat on the mat", ["sat on"]),
        ("Кыргыз тилинде: көңүл, өмүр, жүрөк, таң.", ["өмүр"]),
        ("今天天气很好，我们去公园散步。", ["我们"]),
        ("go \U0001f680 ok \U0001f44d\U0001f3fd dev", ["\U0001f44d"]),
        ("answer\n\nnext question", ["\n\n", "question"]),
    ],
)
def test_detokenizer_stops_before_a_stop_string(
    tiny_tokenizer: Tokenizer, text: str, stop: list[str]
) -> None:
    outputs, stopped = stream(tiny_tokenizer, text, stop)
    assert stopped
    assert "".join(outputs) == expected_output(text, stop)
    # The stop string spans several tokens, so it was held back at least once.
    assert outputs.count("") > 1


def test_detokenizer_without_a_match_returns_all_text(tiny_tokenizer: Tokenizer) -> None:
    text = "the cat sat on the mat"
    outputs, stopped = stream(tiny_tokenizer, text, ["the dog", "mats"])
    assert not stopped
    assert "".join(outputs) == text


def test_detokenizer_finds_a_stop_string_in_text_flushed_at_the_end(
    tiny_tokenizer: Tokenizer,
) -> None:
    # A real U+FFFD is held back as a possible unfinished character and only comes out in
    # finish(), which must still check it against the stop strings.
    outputs, stopped = stream(tiny_tokenizer, "ab�", ["b�"])
    assert stopped
    assert "".join(outputs) == "a"


def test_detokenizer_rejects_an_empty_stop_string(tiny_tokenizer: Tokenizer) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        IncrementalDetokenizer(tiny_tokenizer, stop=[""])
