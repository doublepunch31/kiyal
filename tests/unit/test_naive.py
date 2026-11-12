import dataclasses
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from kiyal.engine.naive import GenerationStep, NaiveGenerator
from kiyal.engine.sequence import FinishReason, SamplingParams
from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model
from kiyal.sampling import SamplingTensors, apply_penalties, token_counts
from kiyal.tokenizer import Tokenizer

CPU = torch.device("cpu")
GREEDY = SamplingParams(max_tokens=24, temperature=0, ignore_eos=True)


@pytest.fixture(scope="module")
def model(tiny_model_dir: Path) -> LlamaModel:
    return load_model(tiny_model_dir, device=CPU, dtype=torch.float32)


@pytest.fixture(scope="module")
def generator(model: LlamaModel, tiny_tokenizer: Tokenizer) -> NaiveGenerator:
    return NaiveGenerator(model, tiny_tokenizer)


@pytest.fixture(scope="module")
def prompt(tiny_tokenizer: Tokenizer) -> list[int]:
    return tiny_tokenizer.encode("the cat sat on")


def tokens_of(steps: list[GenerationStep]) -> list[int]:
    return [step.token_id for step in steps]


def text_of(steps: list[GenerationStep]) -> str:
    return "".join(step.text for step in steps)


def first_new_token_step(token_ids: list[int]) -> int:
    """Index of the first token, after the first step, that did not occur before it."""
    return next(i for i in range(1, len(token_ids)) if token_ids[i] not in token_ids[:i])


@torch.inference_mode()
def test_greedy_generation_matches_an_argmax_loop(
    generator: NaiveGenerator, model: LlamaModel, prompt: list[int]
) -> None:
    ids = torch.tensor([prompt])  # [1, T]
    for _ in range(GREEDY.max_tokens):
        positions = torch.arange(ids.shape[1]).unsqueeze(0)  # [1, T]
        ids = torch.cat([ids, model(ids, positions).argmax(dim=-1, keepdim=True)], dim=1)
    steps = list(generator.generate(prompt, GREEDY))
    assert tokens_of(steps) == ids[0, len(prompt) :].tolist()


def test_generation_ends_at_max_tokens(generator: NaiveGenerator, prompt: list[int]) -> None:
    steps = list(generator.generate(prompt, GREEDY))
    assert len(steps) == GREEDY.max_tokens
    assert [step.finish_reason for step in steps[:-1]] == [None] * (GREEDY.max_tokens - 1)
    assert steps[-1].finish_reason is FinishReason.LENGTH


@pytest.mark.parametrize("params", [GREEDY, SamplingParams(max_tokens=40, seed=3, ignore_eos=True)])
def test_step_texts_add_up_to_the_decoded_output(
    generator: NaiveGenerator, tiny_tokenizer: Tokenizer, prompt: list[int], params: SamplingParams
) -> None:
    steps = list(generator.generate(prompt, params))
    assert text_of(steps) == tiny_tokenizer.decode(tokens_of(steps))


def test_a_stop_token_ends_generation_without_adding_text(
    generator: NaiveGenerator, tiny_tokenizer: Tokenizer, prompt: list[int]
) -> None:
    baseline = tokens_of(list(generator.generate(prompt, GREEDY)))
    stop_step = first_new_token_step(baseline)
    params = dataclasses.replace(GREEDY, stop_token_ids=(baseline[stop_step],))
    steps = list(generator.generate(prompt, params))
    assert tokens_of(steps) == baseline[: stop_step + 1]
    assert steps[-1].finish_reason is FinishReason.STOP
    assert text_of(steps) == tiny_tokenizer.decode(baseline[:stop_step])


def test_an_eos_token_ends_generation_unless_ignored(
    tiny_model_dir: Path, tiny_tokenizer: Tokenizer, prompt: list[int]
) -> None:
    # A separate model instance, so that changing its config does not affect other tests.
    model = load_model(tiny_model_dir, device=CPU, dtype=torch.float32)
    baseline = tokens_of(list(NaiveGenerator(model, tiny_tokenizer).generate(prompt, GREEDY)))
    stop_step = first_new_token_step(baseline)
    model.config = dataclasses.replace(model.config, eos_token_id=(baseline[stop_step],))
    generator = NaiveGenerator(model, tiny_tokenizer)

    steps = list(generator.generate(prompt, dataclasses.replace(GREEDY, ignore_eos=False)))
    assert tokens_of(steps) == baseline[: stop_step + 1]
    assert steps[-1].finish_reason is FinishReason.STOP
    assert tokens_of(list(generator.generate(prompt, GREEDY))) == baseline


def test_the_tokenizer_eos_token_ends_generation(
    tiny_model_dir: Path, tiny_tokenizer: Tokenizer, prompt: list[int]
) -> None:
    model = load_model(tiny_model_dir, device=CPU, dtype=torch.float32)
    baseline = tokens_of(list(NaiveGenerator(model, tiny_tokenizer).generate(prompt, GREEDY)))
    stop_step = first_new_token_step(baseline)
    # config.json names no EOS token here; only the tokenizer does.
    model.config = dataclasses.replace(model.config, eos_token_id=())
    backend = AutoTokenizer.from_pretrained(tiny_model_dir)
    backend.eos_token = backend.convert_ids_to_tokens(baseline[stop_step])
    tokenizer = Tokenizer(backend)
    assert tokenizer.eos_token_id == baseline[stop_step]

    params = dataclasses.replace(GREEDY, ignore_eos=False)
    steps = list(NaiveGenerator(model, tokenizer).generate(prompt, params))
    assert tokens_of(steps) == baseline[: stop_step + 1]
    assert steps[-1].finish_reason is FinishReason.STOP


def test_a_stop_string_ends_generation_before_it(
    generator: NaiveGenerator, prompt: list[int]
) -> None:
    params = SamplingParams(max_tokens=40, seed=3, ignore_eos=True)
    baseline = text_of(list(generator.generate(prompt, params)))
    middle = len(baseline) // 2
    stop = baseline[middle : middle + 3]
    steps = list(generator.generate(prompt, dataclasses.replace(params, stop=(stop,))))
    assert text_of(steps) == baseline[: baseline.index(stop)]
    assert steps[-1].finish_reason is FinishReason.STOP
    assert len(steps) < params.max_tokens


def test_a_seed_reproduces_the_output(generator: NaiveGenerator, prompt: list[int]) -> None:
    def run(seed: int | None) -> list[int]:
        params = SamplingParams(max_tokens=24, seed=seed, ignore_eos=True)
        return tokens_of(list(generator.generate(prompt, params)))

    assert run(1) == run(1)
    assert run(1) != run(2)
    assert run(None) != run(None)


def test_generators_without_a_seed_start_from_different_states(
    model: LlamaModel, tiny_tokenizer: Tokenizer, prompt: list[int]
) -> None:
    params = SamplingParams(max_tokens=24, ignore_eos=True)
    first = list(NaiveGenerator(model, tiny_tokenizer).generate(prompt, params))
    second = list(NaiveGenerator(model, tiny_tokenizer).generate(prompt, params))
    assert tokens_of(first) != tokens_of(second)


@torch.inference_mode()
def penalized_argmax_loop(
    model: LlamaModel,
    params: SamplingParams,
    prompt: list[int],
    counted_as_prompt: bool,
    counted_as_output: bool,
) -> list[int]:
    """Greedy decoding with penalties, with a choice of where the prompt's tokens count."""
    sampling = SamplingTensors.from_params([params], CPU)
    vocab_size = model.config.vocab_size
    output: list[int] = []
    for _ in range(params.max_tokens):
        ids = torch.tensor([prompt + output])  # [1, T]
        logits = model(ids, torch.arange(ids.shape[1]).unsqueeze(0))  # [1, V]
        logits = apply_penalties(
            logits,
            sampling,
            token_counts([prompt if counted_as_prompt else []], vocab_size, CPU),
            token_counts([prompt + output if counted_as_output else output], vocab_size, CPU),
        )
        output.append(int(logits.argmax()))
    return output


def test_penalties_count_prompt_and_output_tokens_separately(
    generator: NaiveGenerator, model: LlamaModel, prompt: list[int]
) -> None:
    # The prompt ends with the start of the model's own greedy continuation, so it holds the
    # tokens the model would pick next and penalties on prompt tokens change the output.
    long_prompt = prompt + tokens_of(list(generator.generate(prompt, GREEDY)))[:12]
    params = dataclasses.replace(GREEDY, repetition_penalty=1.3, frequency_penalty=1.0)

    expected = penalized_argmax_loop(model, params, long_prompt, True, False)
    assert tokens_of(list(generator.generate(long_prompt, params))) == expected
    # The checks above only mean something if the two ways of getting it wrong differ.
    assert expected != penalized_argmax_loop(model, params, long_prompt, False, False)
    assert expected != penalized_argmax_loop(model, params, long_prompt, True, True)
    assert expected != tokens_of(list(generator.generate(long_prompt, GREEDY)))


def test_generation_ends_when_the_context_is_full(
    generator: NaiveGenerator, model: LlamaModel, prompt: list[int]
) -> None:
    max_positions = model.config.max_position_embeddings
    long_prompt = (prompt * max_positions)[: max_positions - 3]
    steps = list(generator.generate(long_prompt, GREEDY))
    assert len(steps) == 3
    assert steps[-1].finish_reason is FinishReason.LENGTH


def test_a_prompt_that_fills_the_context_raises(
    generator: NaiveGenerator, model: LlamaModel, prompt: list[int]
) -> None:
    max_positions = model.config.max_position_embeddings
    with pytest.raises(ValueError, match="at least one output token"):
        next(generator.generate((prompt * max_positions)[:max_positions], GREEDY))


def test_an_empty_prompt_raises(generator: NaiveGenerator) -> None:
    with pytest.raises(ValueError, match="no tokens"):
        next(generator.generate([], GREEDY))


def test_a_tokenizer_larger_than_the_model_raises(
    tiny_checkpoint: Path, tiny_tokenizer: Tokenizer
) -> None:
    small_model = load_model(tiny_checkpoint, device=CPU, dtype=torch.float32)
    with pytest.raises(ValueError, match="the model only has 256"):
        NaiveGenerator(small_model, tiny_tokenizer)


def test_each_step_runs_in_inference_mode_without_leaking_it(
    generator: NaiveGenerator, model: LlamaModel, prompt: list[int]
) -> None:
    in_forward: list[bool] = []
    hook = model.register_forward_pre_hook(
        lambda module, args: in_forward.append(torch.is_inference_mode_enabled())
    )
    steps = generator.generate(prompt, GREEDY)
    next(steps)
    hook.remove()
    assert in_forward == [True]
    # The caller runs between steps and must not inherit the mode.
    assert not torch.is_inference_mode_enabled()
