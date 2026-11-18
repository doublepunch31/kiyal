import dataclasses
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from kiyal.engine.generator import GenerationStep, TextGenerator
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


@pytest.fixture(scope="module", params=[True, False], ids=["cached", "uncached"])
def use_kv_cache(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(scope="module")
def generator(model: LlamaModel, tiny_tokenizer: Tokenizer, use_kv_cache: bool) -> TextGenerator:
    return TextGenerator(model, tiny_tokenizer, use_kv_cache=use_kv_cache)


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
    generator: TextGenerator, model: LlamaModel, prompt: list[int]
) -> None:
    ids = torch.tensor([prompt])  # [1, T]
    for _ in range(GREEDY.max_tokens):
        positions = torch.arange(ids.shape[1]).unsqueeze(0)  # [1, T]
        ids = torch.cat([ids, model(ids, positions).argmax(dim=-1, keepdim=True)], dim=1)
    steps = list(generator.generate(prompt, GREEDY))
    assert tokens_of(steps) == ids[0, len(prompt) :].tolist()


def forward_lengths(model: LlamaModel, generator: TextGenerator, prompt: list[int]) -> list[int]:
    """Number of tokens the model is run on at each step of a greedy generation."""
    lengths: list[int] = []
    hook = model.register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
    list(generator.generate(prompt, GREEDY))
    hook.remove()
    return lengths


def test_the_cache_is_used_by_default(model: LlamaModel, tiny_tokenizer: Tokenizer) -> None:
    prompt = tiny_tokenizer.encode("the cat sat on")
    lengths = forward_lengths(model, TextGenerator(model, tiny_tokenizer), prompt)
    assert lengths == [len(prompt)] + [1] * (GREEDY.max_tokens - 1)


def test_the_model_runs_on_the_tokens_that_are_not_cached(
    generator: TextGenerator, model: LlamaModel, prompt: list[int], use_kv_cache: bool
) -> None:
    lengths = forward_lengths(model, generator, prompt)
    if use_kv_cache:
        # One prefill of the prompt, then one token per step.
        assert lengths == [len(prompt)] + [1] * (GREEDY.max_tokens - 1)
    else:
        assert lengths == [len(prompt) + step for step in range(GREEDY.max_tokens)]


def test_generation_ends_at_max_tokens(generator: TextGenerator, prompt: list[int]) -> None:
    steps = list(generator.generate(prompt, GREEDY))
    assert len(steps) == GREEDY.max_tokens
    assert [step.finish_reason for step in steps[:-1]] == [None] * (GREEDY.max_tokens - 1)
    assert steps[-1].finish_reason is FinishReason.LENGTH


@pytest.mark.parametrize("params", [GREEDY, SamplingParams(max_tokens=40, seed=3, ignore_eos=True)])
def test_step_texts_add_up_to_the_decoded_output(
    generator: TextGenerator, tiny_tokenizer: Tokenizer, prompt: list[int], params: SamplingParams
) -> None:
    steps = list(generator.generate(prompt, params))
    assert text_of(steps) == tiny_tokenizer.decode(tokens_of(steps))


def test_a_stop_token_ends_generation_without_adding_text(
    generator: TextGenerator, tiny_tokenizer: Tokenizer, prompt: list[int]
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
    baseline = tokens_of(list(TextGenerator(model, tiny_tokenizer).generate(prompt, GREEDY)))
    stop_step = first_new_token_step(baseline)
    model.config = dataclasses.replace(model.config, eos_token_id=(baseline[stop_step],))
    generator = TextGenerator(model, tiny_tokenizer)

    steps = list(generator.generate(prompt, dataclasses.replace(GREEDY, ignore_eos=False)))
    assert tokens_of(steps) == baseline[: stop_step + 1]
    assert steps[-1].finish_reason is FinishReason.STOP
    assert tokens_of(list(generator.generate(prompt, GREEDY))) == baseline


def test_the_tokenizer_eos_token_ends_generation(
    tiny_model_dir: Path, tiny_tokenizer: Tokenizer, prompt: list[int]
) -> None:
    model = load_model(tiny_model_dir, device=CPU, dtype=torch.float32)
    baseline = tokens_of(list(TextGenerator(model, tiny_tokenizer).generate(prompt, GREEDY)))
    stop_step = first_new_token_step(baseline)
    # config.json names no EOS token here; only the tokenizer does.
    model.config = dataclasses.replace(model.config, eos_token_id=())
    backend = AutoTokenizer.from_pretrained(tiny_model_dir)
    backend.eos_token = backend.convert_ids_to_tokens(baseline[stop_step])
    tokenizer = Tokenizer(backend)
    assert tokenizer.eos_token_id == baseline[stop_step]

    params = dataclasses.replace(GREEDY, ignore_eos=False)
    steps = list(TextGenerator(model, tokenizer).generate(prompt, params))
    assert tokens_of(steps) == baseline[: stop_step + 1]
    assert steps[-1].finish_reason is FinishReason.STOP


def test_a_stop_string_ends_generation_before_it(
    generator: TextGenerator, prompt: list[int]
) -> None:
    params = SamplingParams(max_tokens=40, seed=3, ignore_eos=True)
    baseline = text_of(list(generator.generate(prompt, params)))
    middle = len(baseline) // 2
    stop = baseline[middle : middle + 3]
    steps = list(generator.generate(prompt, dataclasses.replace(params, stop=(stop,))))
    assert text_of(steps) == baseline[: baseline.index(stop)]
    assert steps[-1].finish_reason is FinishReason.STOP
    assert len(steps) < params.max_tokens


def test_a_seed_reproduces_the_output(generator: TextGenerator, prompt: list[int]) -> None:
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
    first = list(TextGenerator(model, tiny_tokenizer).generate(prompt, params))
    second = list(TextGenerator(model, tiny_tokenizer).generate(prompt, params))
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
    generator: TextGenerator, model: LlamaModel, prompt: list[int]
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
    generator: TextGenerator, model: LlamaModel, prompt: list[int]
) -> None:
    max_positions = model.config.max_position_embeddings
    long_prompt = (prompt * max_positions)[: max_positions - 3]
    steps = list(generator.generate(long_prompt, GREEDY))
    assert len(steps) == 3
    assert steps[-1].finish_reason is FinishReason.LENGTH


def test_a_prompt_that_fills_the_context_raises(
    generator: TextGenerator, model: LlamaModel, prompt: list[int]
) -> None:
    max_positions = model.config.max_position_embeddings
    with pytest.raises(ValueError, match="at least one output token"):
        next(generator.generate((prompt * max_positions)[:max_positions], GREEDY))


def test_an_empty_prompt_raises(generator: TextGenerator) -> None:
    with pytest.raises(ValueError, match="no tokens"):
        next(generator.generate([], GREEDY))


def test_a_tokenizer_larger_than_the_model_raises(
    tiny_checkpoint: Path, tiny_tokenizer: Tokenizer
) -> None:
    small_model = load_model(tiny_checkpoint, device=CPU, dtype=torch.float32)
    with pytest.raises(ValueError, match="the model only has 256"):
        TextGenerator(small_model, tiny_tokenizer)


def test_each_step_runs_in_inference_mode_without_leaking_it(
    generator: TextGenerator, model: LlamaModel, prompt: list[int]
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


BATCH_TEXTS = ["the cat sat on", "мама мыла раму", "你好", "кыял деген эмне, кыял"]


@pytest.fixture(scope="module")
def batch_prompts(tiny_tokenizer: Tokenizer) -> list[list[int]]:
    prompts = [tiny_tokenizer.encode(text) for text in BATCH_TEXTS]
    # Different lengths, so that every row but the longest starts with padding.
    assert len({len(prompt) for prompt in prompts}) == len(prompts)
    return prompts


def row_steps(batch: list[list[GenerationStep | None]], row: int) -> list[GenerationStep | None]:
    return [steps[row] for steps in batch]


def test_each_row_of_a_batch_generates_what_it_generates_alone(
    generator: TextGenerator, batch_prompts: list[list[int]]
) -> None:
    params = [
        GREEDY,
        dataclasses.replace(GREEDY, max_tokens=7),
        SamplingParams(max_tokens=30, seed=5, ignore_eos=True),
        dataclasses.replace(GREEDY, max_tokens=16, repetition_penalty=1.3, frequency_penalty=1.0),
    ]
    batch = list(generator.generate_batch(batch_prompts, params))
    alone = [list(generator.generate(p, sp)) for p, sp in zip(batch_prompts, params, strict=True)]
    assert len(batch) == max(len(steps) for steps in alone)
    for row, expected in enumerate(alone):
        # A finished row has no step until the batch ends.
        padding = [None] * (len(batch) - len(expected))
        assert row_steps(batch, row) == [*expected, *padding]


def test_the_batch_ends_with_its_last_output(
    generator: TextGenerator, batch_prompts: list[list[int]]
) -> None:
    params = [dataclasses.replace(GREEDY, max_tokens=n) for n in (3, 6)]
    batch = list(generator.generate_batch(batch_prompts[:2], params))
    assert len(batch) == 6
    first, second = row_steps(batch, 0), row_steps(batch, 1)
    assert first[3:] == [None, None, None]
    assert [step and step.finish_reason for step in first[:3]] == [None, None, FinishReason.LENGTH]
    assert all(step is not None for step in second)
    assert second[-1] is not None and second[-1].finish_reason is FinishReason.LENGTH


def test_a_batch_runs_the_model_on_the_padded_prompts_then_one_token_per_row(
    generator: TextGenerator, model: LlamaModel, batch_prompts: list[list[int]], use_kv_cache: bool
) -> None:
    shapes: list[tuple[int, int]] = []
    hook = model.register_forward_pre_hook(lambda module, args: shapes.append(args[0].shape))
    list(generator.generate_batch(batch_prompts, [GREEDY] * len(batch_prompts)))
    hook.remove()
    longest = max(len(prompt) for prompt in batch_prompts)
    rows = len(batch_prompts)
    if use_kv_cache:
        assert shapes == [(rows, longest)] + [(rows, 1)] * (GREEDY.max_tokens - 1)
    else:
        assert shapes == [(rows, longest + step) for step in range(GREEDY.max_tokens)]


def test_a_row_that_fills_the_context_ends_while_the_others_continue(
    generator: TextGenerator, model: LlamaModel, prompt: list[int]
) -> None:
    # The finished row keeps running through the model, past the end of its context.
    max_positions = model.config.max_position_embeddings
    long_prompt = (prompt * max_positions)[: max_positions - 3]
    batch = list(generator.generate_batch([long_prompt, prompt], [GREEDY, GREEDY]))
    assert len(batch) == GREEDY.max_tokens
    assert row_steps(batch, 0) == [
        *generator.generate(long_prompt, GREEDY),
        *[None] * (GREEDY.max_tokens - 3),
    ]
    assert row_steps(batch, 1) == list(generator.generate(prompt, GREEDY))


def test_invalid_batches_raise(generator: TextGenerator, prompt: list[int]) -> None:
    with pytest.raises(ValueError, match="no prompts"):
        next(generator.generate_batch([], []))
    with pytest.raises(ValueError, match="2 prompts but 1 sampling settings"):
        next(generator.generate_batch([prompt, prompt], [GREEDY]))
    with pytest.raises(ValueError, match="prompt 1 has no tokens"):
        next(generator.generate_batch([prompt, []], [GREEDY, GREEDY]))
