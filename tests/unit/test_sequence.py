import dataclasses
from typing import Any

import pytest

from kiyal.engine.sequence import (
    FinishReason,
    SamplingParams,
    SequenceFactory,
    SequenceStatus,
)
from kiyal.model.config import ModelConfig
from kiyal.tokenizer import Tokenizer


def test_defaults_sample_from_the_unfiltered_distribution() -> None:
    params = SamplingParams()
    assert (params.temperature, params.top_k, params.top_p, params.min_p) == (1.0, 0, 1.0, 0.0)
    penalties = (params.repetition_penalty, params.presence_penalty, params.frequency_penalty)
    assert penalties == (1.0, 0.0, 0.0)
    assert params.seed is None
    assert (params.stop, params.stop_token_ids, params.ignore_eos) == ((), (), False)


@pytest.mark.parametrize(
    "settings",
    [
        {"temperature": 0.0},
        {"temperature": 2.5},
        {"top_k": 1},
        {"top_p": 1e-6},
        {"min_p": 1.0},
        {"repetition_penalty": 0.5},
        {"presence_penalty": -2.0},
        {"frequency_penalty": 2.0},
        {"seed": 0},
        {"seed": 2**63 - 1},
        {"max_tokens": 1},
        {"stop": ("\n\n", "END")},
        {"stop_token_ids": (0, 7)},
    ],
)
def test_values_at_the_edge_of_their_range_are_accepted(settings: dict[str, Any]) -> None:
    SamplingParams(**settings)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": float("nan")}, "temperature"),
        ({"temperature": float("inf")}, "temperature"),
        ({"top_k": -1}, "top_k"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
        ({"top_p": float("nan")}, "top_p"),
        ({"min_p": -0.1}, "min_p"),
        ({"min_p": 1.1}, "min_p"),
        ({"repetition_penalty": 0.0}, "repetition_penalty"),
        ({"repetition_penalty": float("inf")}, "repetition_penalty"),
        ({"presence_penalty": 2.1}, "presence_penalty"),
        ({"presence_penalty": float("nan")}, "presence_penalty"),
        ({"frequency_penalty": -2.1}, "frequency_penalty"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**63}, "seed"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"stop": ("END", "")}, "stop"),
        ({"stop": "END"}, "stop"),
        ({"stop_token_ids": (3, -1)}, "stop_token_ids"),
    ],
)
def test_values_out_of_range_raise(settings: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SamplingParams(**settings)


MAX_LEN = 12


@pytest.fixture
def config(tiny_config: ModelConfig, tiny_tokenizer: Tokenizer) -> ModelConfig:
    """The tiny model's config, sized for the tiny tokenizer, with EOS token 2."""
    return dataclasses.replace(tiny_config, vocab_size=tiny_tokenizer.vocab_size)


@pytest.fixture
def factory(config: ModelConfig, tiny_tokenizer: Tokenizer) -> SequenceFactory:
    return SequenceFactory(config, tiny_tokenizer, MAX_LEN)


def test_a_new_sequence_waits_with_nothing_computed(factory: SequenceFactory) -> None:
    seq = factory.create("a", [5, 6, 7], SamplingParams(max_tokens=4))
    assert seq.request_id == "a"
    assert seq.status is SequenceStatus.WAITING
    assert (seq.num_tokens, seq.num_computed_tokens, seq.block_table) == (3, 0, [])
    assert seq.uncomputed_token_ids() == [5, 6, 7]
    assert seq.finish_reason is None
    assert seq.generator is None
    assert seq.arrival_time > 0


def test_max_tokens_is_lowered_to_fit_the_context(factory: SequenceFactory) -> None:
    assert factory.create("a", [5] * 4, SamplingParams(max_tokens=4)).max_tokens == 4
    assert factory.create("b", [5] * 4, SamplingParams(max_tokens=100)).max_tokens == MAX_LEN - 4


def test_eos_tokens_stop_the_output_unless_ignored(
    factory: SequenceFactory, config: ModelConfig, tiny_tokenizer: Tokenizer
) -> None:
    eos = {*config.eos_token_id, tiny_tokenizer.eos_token_id}
    seq = factory.create("a", [5], SamplingParams(stop_token_ids=(9,)))
    assert seq.stop_token_ids == {9, *eos}
    seq = factory.create("b", [5], SamplingParams(stop_token_ids=(9,), ignore_eos=True))
    assert seq.stop_token_ids == {9}


def test_a_seed_gives_the_sequence_its_own_generator(factory: SequenceFactory) -> None:
    first = factory.create("a", [5], SamplingParams(seed=3))
    second = factory.create("b", [5], SamplingParams(seed=3))
    assert first.generator is not None and second.generator is not None
    assert first.generator is not second.generator
    assert first.generator.initial_seed() == 3


def test_only_the_latest_token_is_uncomputed_after_the_prompt_ran(
    factory: SequenceFactory,
) -> None:
    seq = factory.create("a", [5, 6, 7], SamplingParams(max_tokens=4))
    seq.num_computed_tokens = 3
    seq.append(8)
    assert seq.uncomputed_token_ids() == [8]
    seq.num_computed_tokens = 4
    seq.append(9)
    assert seq.uncomputed_token_ids() == [9]
    assert seq.num_tokens == 5
    # Before the prompt has run, everything is uncomputed.
    seq.num_computed_tokens = 0
    assert seq.uncomputed_token_ids() == [5, 6, 7, 8, 9]


def test_the_output_ends_at_max_tokens(factory: SequenceFactory) -> None:
    seq = factory.create("a", [5], SamplingParams(max_tokens=2, ignore_eos=True))
    seq.append(6)
    assert not seq.is_finished
    assert seq.first_token_time is not None
    seq.append(7)
    assert seq.is_finished
    assert seq.status is SequenceStatus.FINISHED
    assert seq.finish_reason is FinishReason.LENGTH
    assert seq.finished_time is not None
    assert seq.last_token_time is not None
    assert seq.first_token_time <= seq.last_token_time <= seq.finished_time


def test_a_stop_token_ends_the_output_without_text(factory: SequenceFactory) -> None:
    seq = factory.create("a", [5], SamplingParams(max_tokens=8, stop_token_ids=(9,)))
    assert seq.append(9) == ""
    assert seq.finish_reason is FinishReason.STOP


def test_finish_records_the_reason(factory: SequenceFactory) -> None:
    seq = factory.create("a", [5], SamplingParams())
    seq.finish(FinishReason.ABORT)
    assert seq.is_finished
    assert seq.finish_reason is FinishReason.ABORT


@pytest.mark.parametrize(
    ("prompt", "message"),
    [
        ([], "the prompt has no tokens"),
        ([5] * MAX_LEN, "has 12 tokens, but the context of 12 tokens must also fit"),
    ],
)
def test_prompts_without_room_for_output_raise(
    factory: SequenceFactory, prompt: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory.create("a", prompt, SamplingParams())


def test_the_prompt_can_be_named_in_errors(factory: SequenceFactory) -> None:
    with pytest.raises(ValueError, match="prompt 3 has no tokens"):
        factory.create("a", [], SamplingParams(), name="prompt 3")


@pytest.mark.parametrize("max_len", [1, 129])
def test_a_max_len_outside_the_context_raises(
    config: ModelConfig, tiny_tokenizer: Tokenizer, max_len: int
) -> None:
    with pytest.raises(ValueError, match=r"max_len must be in \[2, 128\]"):
        SequenceFactory(config, tiny_tokenizer, max_len)


def test_a_tokenizer_larger_than_the_model_raises(
    tiny_config: ModelConfig, tiny_tokenizer: Tokenizer
) -> None:
    with pytest.raises(ValueError, match="but the model only has 256"):
        SequenceFactory(tiny_config, tiny_tokenizer, MAX_LEN)
