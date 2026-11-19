from pathlib import Path

import pytest
import torch
from torch import Tensor

from kiyal.cache.contiguous import ContiguousKVPool
from kiyal.config import EngineConfig
from kiyal.engine.generator import TextGenerator
from kiyal.engine.llm_engine import LLMEngine
from kiyal.engine.model_runner import ModelRunner
from kiyal.engine.sequence import FinishReason, RequestOutput, SamplingParams, SequenceFactory
from kiyal.model.attention import AttentionMetadata
from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model
from kiyal.tokenizer import Tokenizer

CPU = torch.device("cpu")
GREEDY = SamplingParams(max_tokens=12, temperature=0, ignore_eos=True)
TEXTS = ["the cat sat on", "мама мыла раму", "кыял деген эмне", "你好世界"]


@pytest.fixture(scope="module")
def model(tiny_model_dir: Path) -> LlamaModel:
    return load_model(tiny_model_dir, device=CPU, dtype=torch.float32)


@pytest.fixture
def engine(model: LlamaModel, tiny_tokenizer: Tokenizer) -> LLMEngine:
    return LLMEngine(model, tiny_tokenizer, EngineConfig(max_num_seqs=4))


@pytest.fixture(scope="module")
def prompts(tiny_tokenizer: Tokenizer) -> list[list[int]]:
    return [tiny_tokenizer.encode(text) for text in TEXTS]


def run_to_end(engine: LLMEngine) -> dict[str, list[RequestOutput]]:
    outputs: dict[str, list[RequestOutput]] = {}
    while engine.has_unfinished():
        for output in engine.step():
            outputs.setdefault(output.request_id, []).append(output)
    return outputs


def test_the_runner_returns_each_sequences_next_token_logits(
    model: LlamaModel, tiny_tokenizer: Tokenizer, prompts: list[list[int]]
) -> None:
    factory = SequenceFactory(model.config, tiny_tokenizer, 32)
    runner = ModelRunner(model, num_rows=4, max_len=32)
    sequences = [factory.create(str(i), prompt, GREEDY) for i, prompt in enumerate(prompts)]
    for row, seq in zip([3, 1, 0, 2], sequences, strict=True):
        seq.cache_row = row
    logits = runner.execute(sequences)
    assert logits.shape == (len(prompts), model.config.vocab_size)
    with torch.inference_mode():
        for row_logits, prompt in zip(logits, prompts, strict=True):
            alone = model(torch.tensor([prompt]), torch.arange(len(prompt)).unsqueeze(0))[0]
            torch.testing.assert_close(row_logits, alone, rtol=1e-5, atol=1e-5)


def test_a_request_alone_generates_what_the_static_generator_does(
    engine: LLMEngine, model: LlamaModel, tiny_tokenizer: Tokenizer, prompts: list[list[int]]
) -> None:
    static = TextGenerator(model, tiny_tokenizer)
    for prompt in prompts:
        expected = list(static.generate(prompt, GREEDY))
        request_id = engine.add_request(prompt, GREEDY)
        outputs = run_to_end(engine)[request_id]
        assert [(o.token_id, o.text, o.finish_reason) for o in outputs] == [
            (s.token_id, s.text, s.finish_reason) for s in expected
        ]


def test_requests_in_one_batch_generate_what_they_generate_alone(
    engine: LLMEngine, prompts: list[list[int]]
) -> None:
    alone = [engine.generate([prompt], [GREEDY])[0] for prompt in prompts]
    together = engine.generate(prompts, [GREEDY] * len(prompts))
    assert [c.token_ids for c in together] == [c.token_ids for c in alone]
    assert [c.text for c in together] == [c.text for c in alone]


def test_a_step_gives_one_output_per_request_in_it(
    engine: LLMEngine, prompts: list[list[int]]
) -> None:
    first = engine.add_request(prompts[0], GREEDY)
    second = engine.add_request(prompts[1], GREEDY)
    assert [o.request_id for o in engine.step()] == [first, second]
    assert [o.request_id for o in engine.step()] == [first, second]


def test_a_new_request_joins_the_running_batch(engine: LLMEngine, prompts: list[list[int]]) -> None:
    first = engine.add_request(prompts[0], GREEDY)
    for _ in range(3):
        engine.step()
    second = engine.add_request(prompts[1], GREEDY)
    # The new prompt runs alone, then both decode together.
    assert [o.request_id for o in engine.step()] == [second]
    assert [o.request_id for o in engine.step()] == [first, second]


def test_a_finished_request_leaves_the_batch(engine: LLMEngine, prompts: list[list[int]]) -> None:
    short = engine.add_request(
        prompts[0], SamplingParams(max_tokens=2, temperature=0, ignore_eos=True)
    )
    long = engine.add_request(prompts[1], GREEDY)
    engine.step()
    last = engine.step()
    assert [(o.request_id, o.finish_reason) for o in last] == [
        (short, FinishReason.LENGTH),
        (long, None),
    ]
    assert [o.request_id for o in engine.step()] == [long]


def test_a_waiting_request_starts_when_a_running_one_ends(
    model: LlamaModel, tiny_tokenizer: Tokenizer, prompts: list[list[int]]
) -> None:
    engine = LLMEngine(model, tiny_tokenizer, EngineConfig(max_num_seqs=1))
    first = engine.add_request(
        prompts[0], SamplingParams(max_tokens=2, temperature=0, ignore_eos=True)
    )
    second = engine.add_request(prompts[1], GREEDY)
    assert [o.request_id for o in engine.step()] == [first]
    assert [o.request_id for o in engine.step()] == [first]
    assert [o.request_id for o in engine.step()] == [second]


def forward_sizes(
    model: LlamaModel, engine: LLMEngine, monkeypatch: pytest.MonkeyPatch
) -> list[int]:
    """Runs the engine to the end and returns the number of tokens of every forward pass."""
    sizes: list[int] = []
    forward_flat = model.forward_flat

    def record(
        input_ids: Tensor,
        positions: Tensor,
        pool: ContiguousKVPool,
        metadata: AttentionMetadata,
        logits_indices: Tensor,
    ) -> Tensor:
        sizes.append(len(input_ids))
        return forward_flat(input_ids, positions, pool, metadata, logits_indices)

    monkeypatch.setattr(model, "forward_flat", record)
    run_to_end(engine)
    return sizes


def test_the_model_runs_on_the_tokens_of_unfinished_requests_only(
    engine: LLMEngine, model: LlamaModel, prompts: list[list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    engine.add_request(prompts[0], SamplingParams(max_tokens=2, temperature=0, ignore_eos=True))
    engine.add_request(prompts[1], SamplingParams(max_tokens=4, temperature=0, ignore_eos=True))
    sizes = forward_sizes(model, engine, monkeypatch)
    # Both prompts without padding, a decode of both, then the second alone.
    assert sizes == [len(prompts[0]) + len(prompts[1]), 2, 1, 1]


def test_generate_returns_each_prompts_completion(
    engine: LLMEngine, model: LlamaModel, tiny_tokenizer: Tokenizer, prompts: list[list[int]]
) -> None:
    params = [SamplingParams(max_tokens=n, temperature=0) for n in (5, 9, 1, 7)]
    completions = engine.generate(prompts, params)
    static = TextGenerator(model, tiny_tokenizer)
    for completion, prompt, prompt_params in zip(completions, prompts, params, strict=True):
        steps = list(static.generate(prompt, prompt_params))
        assert completion.token_ids == [step.token_id for step in steps]
        assert completion.text == "".join(step.text for step in steps)
        assert completion.finish_reason == steps[-1].finish_reason
    assert not engine.has_unfinished()


def test_generate_needs_an_idle_engine(engine: LLMEngine, prompts: list[list[int]]) -> None:
    engine.add_request(prompts[0], GREEDY)
    with pytest.raises(RuntimeError, match="without unfinished requests"):
        engine.generate(prompts, [GREEDY] * len(prompts))


def test_generate_adds_nothing_when_a_prompt_is_invalid(
    engine: LLMEngine, prompts: list[list[int]]
) -> None:
    with pytest.raises(ValueError, match="the prompt has no tokens"):
        engine.generate([prompts[0], []], [GREEDY, GREEDY])
    assert not engine.has_unfinished()
    with pytest.raises(ValueError, match="got 1 prompts but 2 sampling settings"):
        engine.generate([prompts[0]], [GREEDY, GREEDY])


def test_aborting_a_request_frees_its_place(
    model: LlamaModel, tiny_tokenizer: Tokenizer, prompts: list[list[int]]
) -> None:
    engine = LLMEngine(model, tiny_tokenizer, EngineConfig(max_num_seqs=1))
    first = engine.add_request(prompts[0], GREEDY)
    second = engine.add_request(prompts[1], GREEDY)
    engine.step()
    engine.abort(first)
    assert [o.request_id for o in engine.step()] == [second]
    # Aborting again, or an unknown request, does nothing.
    engine.abort(first)
    engine.abort("unknown")
    engine.abort(second)
    assert not engine.has_unfinished()
    assert engine.step() == []


def test_request_ids_are_unique(engine: LLMEngine, prompts: list[list[int]]) -> None:
    assert engine.add_request(prompts[0], GREEDY, request_id="1") == "1"
    ids = {engine.add_request(prompts[0], GREEDY) for _ in range(3)}
    assert len(ids) == 3
    assert "1" not in ids
    with pytest.raises(ValueError, match="request 1 is already running"):
        engine.add_request(prompts[0], GREEDY, request_id="1")
    # The id of a finished request can be used again.
    run_to_end(engine)
    assert engine.add_request(prompts[0], GREEDY, request_id="1") == "1"


def test_prompts_that_cannot_run_raise(
    model: LlamaModel, tiny_tokenizer: Tokenizer, prompts: list[list[int]]
) -> None:
    config = EngineConfig(max_num_seqs=2, max_num_batched_tokens=16, max_model_len=12)
    engine = LLMEngine(model, tiny_tokenizer, config)
    with pytest.raises(ValueError, match="the prompt has no tokens"):
        engine.add_request([], GREEDY)
    with pytest.raises(ValueError, match="but the context of 12 tokens must also fit"):
        engine.add_request([5] * 12, GREEDY)
    engine = LLMEngine(
        model, tiny_tokenizer, EngineConfig(max_num_seqs=2, max_num_batched_tokens=8)
    )
    with pytest.raises(ValueError, match="more than the 8 that one step can run"):
        engine.add_request([5] * 9, GREEDY)
    assert not engine.has_unfinished()


def test_max_model_len_is_lowered_to_the_models_context(
    model: LlamaModel, tiny_tokenizer: Tokenizer
) -> None:
    context = model.config.max_position_embeddings
    engine = LLMEngine(
        model, tiny_tokenizer, EngineConfig(max_num_seqs=1, max_num_batched_tokens=4 * context)
    )
    with pytest.raises(ValueError, match=f"but the context of {context} tokens"):
        engine.add_request([5] * context, GREEDY)
    # A prompt that leaves one position in the context gets one token.
    request_id = engine.add_request([5] * (context - 1), SamplingParams(max_tokens=10))
    outputs = run_to_end(engine)[request_id]
    assert len(outputs) == 1
    assert outputs[0].finish_reason is FinishReason.LENGTH


def test_a_seed_gives_the_same_sample_alone_and_in_a_batch(
    engine: LLMEngine, prompts: list[list[int]]
) -> None:
    seeded = SamplingParams(max_tokens=12, temperature=1.0, seed=7, ignore_eos=True)
    alone = engine.generate([prompts[0]], [seeded])[0]
    others = SamplingParams(max_tokens=12, temperature=1.0, ignore_eos=True)
    together = engine.generate(prompts, [seeded, others, others, others])[0]
    assert together.token_ids == alone.token_ids


def test_penalties_apply_per_request(engine: LLMEngine, prompts: list[list[int]]) -> None:
    penalized = SamplingParams(max_tokens=12, temperature=0, ignore_eos=True, frequency_penalty=2.0)
    alone = engine.generate([prompts[0]], [penalized])[0]
    together = engine.generate(prompts[:2], [penalized, GREEDY])
    assert together[0].token_ids == alone.token_ids
    assert together[1].token_ids == engine.generate([prompts[1]], [GREEDY])[0].token_ids
    assert alone.token_ids != engine.generate([prompts[0]], [GREEDY])[0].token_ids


def test_each_step_runs_in_inference_mode(engine: LLMEngine, prompts: list[list[int]]) -> None:
    engine.add_request(prompts[0], GREEDY)
    engine.step()
    assert not torch.is_inference_mode_enabled()
