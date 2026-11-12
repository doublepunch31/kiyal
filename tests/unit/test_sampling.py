import math

import pytest
import torch
from torch import Tensor

from kiyal.engine.sequence import SamplingParams
from kiyal.sampling import (
    SamplingTensors,
    apply_penalties,
    draw_uniform,
    sample,
    token_counts,
    token_probs,
)

CPU = torch.device("cpu")
DRAWS = 20_000
# The standard error of a frequency over 20,000 draws is at most 0.0036, so this is 4 of them.
FREQUENCY_TOLERANCE = 0.015


def batch(*params: SamplingParams) -> SamplingTensors:
    return SamplingTensors.from_params(params, CPU)


def probs_for(probs: list[float], params: SamplingParams) -> Tensor:
    """Filters one row whose unfiltered distribution is `probs`."""
    logits = torch.tensor(probs).log().unsqueeze(0)  # [1, V]
    return token_probs(logits, batch(params))[0]  # [V]


def sampled_frequencies(logits: Tensor, params: SamplingParams, seed: int = 0) -> Tensor:
    """Samples one row's logits [V] many times and returns how often each token came up."""
    uniform = torch.rand(DRAWS, generator=torch.Generator().manual_seed(seed))
    tokens = sample(logits.expand(DRAWS, -1), batch(*[params] * DRAWS), uniform)
    return torch.bincount(tokens, minlength=logits.shape[0]) / DRAWS  # [V]


def reference_probs(logits: list[float], params: SamplingParams) -> list[float]:
    """The filters written one token at a time, renormalizing after each step."""

    def normalized(weights: dict[int, float]) -> dict[int, float]:
        total = sum(weights.values())
        return {token: weight / total for token, weight in weights.items()}

    scaled = [x / params.temperature for x in logits]
    order = sorted(range(len(scaled)), key=lambda token: -scaled[token])
    kept = order[: params.top_k] if params.top_k > 0 else order
    probs = normalized({token: math.exp(scaled[token] - scaled[order[0]]) for token in kept})

    nucleus: dict[int, float] = {}
    for token in kept:
        if params.top_p < 1 and sum(nucleus.values()) >= params.top_p:
            break
        nucleus[token] = probs[token]
    probs = normalized(nucleus)

    floor = params.min_p * probs[order[0]]
    probs = normalized({token: p for token, p in probs.items() if p >= floor})
    return [probs.get(token, 0.0) for token in range(len(logits))]


def test_temperature_zero_picks_the_most_likely_token() -> None:
    logits = torch.randn(8, 50, generator=torch.Generator().manual_seed(0))
    tokens = sample(logits, batch(*[SamplingParams(temperature=0)] * 8), torch.rand(8))
    assert torch.equal(tokens, logits.argmax(dim=-1))
    probs = token_probs(logits, batch(*[SamplingParams(temperature=0)] * 8))
    assert torch.equal(probs, torch.nn.functional.one_hot(tokens, 50).float())


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_sampled_frequencies_follow_the_temperature(temperature: float) -> None:
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0, 0.5])
    frequencies = sampled_frequencies(logits, SamplingParams(temperature=temperature))
    expected = torch.softmax(logits / temperature, dim=-1)
    torch.testing.assert_close(frequencies, expected, atol=FREQUENCY_TOLERANCE, rtol=0)


def test_evenly_spaced_draws_give_each_token_its_share() -> None:
    # With draws on an even grid instead of random ones, each token must come up in exact
    # proportion to its probability, up to one grid step at each interval's ends.
    probs = torch.tensor([0.05, 0.4, 0.25, 0.3])
    uniform = (torch.arange(DRAWS) + 0.5) / DRAWS
    tokens = sample(probs.log().expand(DRAWS, -1), batch(*[SamplingParams()] * DRAWS), uniform)
    frequencies = torch.bincount(tokens, minlength=4) / DRAWS
    torch.testing.assert_close(frequencies, probs, atol=2 / DRAWS, rtol=0)


def test_rows_keep_their_own_settings() -> None:
    logits = torch.tensor([0.0, 1.0, 3.0, 2.0])
    settings = [SamplingParams(temperature=0), SamplingParams(temperature=0.8)]
    uniform = torch.rand(DRAWS, generator=torch.Generator().manual_seed(1))
    tokens = sample(logits.expand(DRAWS, -1), batch(*settings * (DRAWS // 2)), uniform)

    greedy_rows, sampled_rows = tokens[0::2], tokens[1::2]
    assert torch.all(greedy_rows == 2)
    frequencies = torch.bincount(sampled_rows, minlength=4) / len(sampled_rows)
    expected = torch.softmax(logits / 0.8, dim=-1)
    # Half as many draws, so the standard error is larger by a factor of sqrt(2).
    torch.testing.assert_close(frequencies, expected, atol=1.5 * FREQUENCY_TOLERANCE, rtol=0)


def test_top_k_keeps_the_k_most_likely_tokens() -> None:
    probs = probs_for([0.1, 0.4, 0.2, 0.3], SamplingParams(top_k=2))
    torch.testing.assert_close(probs, torch.tensor([0.0, 4 / 7, 0.0, 3 / 7]))

    frequencies = sampled_frequencies(
        torch.tensor([0.1, 0.4, 0.2, 0.3]).log(), SamplingParams(top_k=2)
    )
    assert frequencies[0] == 0
    assert frequencies[2] == 0
    torch.testing.assert_close(frequencies, probs, atol=FREQUENCY_TOLERANCE, rtol=0)


def test_top_k_of_one_is_greedy() -> None:
    logits = torch.randn(DRAWS // 100, 30, generator=torch.Generator().manual_seed(2))
    uniform = torch.rand(len(logits), generator=torch.Generator().manual_seed(3))
    tokens = sample(logits, batch(*[SamplingParams(top_k=1)] * len(logits)), uniform)
    assert torch.equal(tokens, logits.argmax(dim=-1))


@pytest.mark.parametrize("top_k", [0, 4, 100])
def test_top_k_of_the_vocabulary_or_more_keeps_everything(top_k: int) -> None:
    probs = [0.1, 0.4, 0.2, 0.3]
    filtered = probs_for(probs, SamplingParams(top_k=top_k))
    torch.testing.assert_close(filtered, torch.tensor(probs))


@pytest.mark.parametrize(
    ("top_p", "expected"),
    [
        (0.45, [0.0, 1.0, 0.0, 0.0]),
        (0.7, [0.0, 0.625, 0.0, 0.375]),
        (0.81, [0.15 / 0.95, 0.5 / 0.95, 0.0, 0.3 / 0.95]),
        (0.96, [0.15, 0.5, 0.05, 0.3]),
    ],
)
def test_top_p_keeps_the_smallest_set_reaching_its_mass(
    top_p: float, expected: list[float]
) -> None:
    probs = probs_for([0.15, 0.5, 0.05, 0.3], SamplingParams(top_p=top_p))
    torch.testing.assert_close(probs, torch.tensor(expected))


def test_top_p_of_one_keeps_every_token() -> None:
    # Spread-out logits give a long tail of tiny probabilities, where the running sum has
    # already rounded to 1.
    logits = 4 * torch.randn(3, 50_000, generator=torch.Generator().manual_seed(4))
    probs = token_probs(logits, batch(*[SamplingParams(top_p=1.0)] * 3))
    expected = torch.softmax(logits, dim=-1)
    torch.testing.assert_close(probs, expected)
    assert torch.equal(probs > 0, expected > 0)


@pytest.mark.parametrize(
    ("min_p", "expected"),
    [
        (0.0, [0.15, 0.5, 0.05, 0.3]),
        (0.2, [0.15 / 0.95, 0.5 / 0.95, 0.0, 0.3 / 0.95]),
        (0.5, [0.0, 0.625, 0.0, 0.375]),
        (1.0, [0.0, 1.0, 0.0, 0.0]),
    ],
)
def test_min_p_drops_tokens_far_below_the_most_likely(min_p: float, expected: list[float]) -> None:
    probs = probs_for([0.15, 0.5, 0.05, 0.3], SamplingParams(min_p=min_p))
    torch.testing.assert_close(probs, torch.tensor(expected))


def test_top_p_sees_the_distribution_left_by_top_k() -> None:
    # After top-k the two survivors hold 4/7 and 3/7, so the first alone reaches 0.5.
    # Applied to the original distribution, top-p would keep both.
    probs = probs_for([0.4, 0.3, 0.2, 0.1], SamplingParams(top_k=2, top_p=0.5))
    torch.testing.assert_close(probs, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_min_p_runs_after_top_p() -> None:
    # Top-p keeps 0.4 and 0.3, and both pass min-p. In the other order, min-p would leave
    # 4/7 and 3/7 and top-p would then drop the second.
    probs = probs_for([0.4, 0.3, 0.2, 0.1], SamplingParams(top_p=0.5, min_p=0.6))
    torch.testing.assert_close(probs, torch.tensor([4 / 7, 3 / 7, 0.0, 0.0]))


def test_top_p_sees_the_distribution_left_by_temperature() -> None:
    # At temperature 1 the first token alone reaches 0.6. Temperature 3 flattens the
    # distribution to about [0.46, 0.30, 0.24], so the nucleus needs two tokens.
    cold = probs_for([0.7, 0.2, 0.1], SamplingParams(top_p=0.6))
    hot = probs_for([0.7, 0.2, 0.1], SamplingParams(top_p=0.6, temperature=3.0))
    torch.testing.assert_close(cold, torch.tensor([1.0, 0.0, 0.0]))
    weights = torch.tensor([0.7, 0.2]) ** (1 / 3)
    torch.testing.assert_close(hot[:2], weights / weights.sum())
    assert hot[2] == 0


def test_filters_match_a_token_by_token_reference() -> None:
    generator = torch.Generator().manual_seed(5)
    settings = [
        SamplingParams(
            temperature=float(torch.empty(()).uniform_(0.3, 2.0, generator=generator)),
            top_k=int(torch.randint(0, 25, (), generator=generator)),
            top_p=float(torch.empty(()).uniform_(0.1, 1.0, generator=generator)),
            min_p=float(torch.empty(()).uniform_(0.0, 0.5, generator=generator)),
        )
        for _ in range(200)
    ]
    logits = 2 * torch.randn(len(settings), 20, generator=generator)
    expected = [
        reference_probs(row, params) for row, params in zip(logits.tolist(), settings, strict=True)
    ]
    torch.testing.assert_close(token_probs(logits, batch(*settings)), torch.tensor(expected))


def test_equal_logits_are_sampled_uniformly() -> None:
    logits = torch.zeros(5)
    torch.testing.assert_close(probs_for([0.2] * 5, SamplingParams()), torch.full((5,), 0.2))
    frequencies = sampled_frequencies(logits, SamplingParams())
    torch.testing.assert_close(frequencies, torch.full((5,), 0.2), atol=FREQUENCY_TOLERANCE, rtol=0)


def test_equal_logits_break_ties_by_token_id() -> None:
    logits = torch.zeros(1, 5)
    greedy = sample(logits, batch(SamplingParams(temperature=0)), torch.rand(1))
    assert greedy.tolist() == [0]
    torch.testing.assert_close(
        probs_for([0.2] * 5, SamplingParams(top_k=1)), torch.tensor([1.0, 0, 0, 0, 0])
    )
    torch.testing.assert_close(
        probs_for([0.2] * 5, SamplingParams(top_k=2)), torch.tensor([0.5, 0.5, 0, 0, 0])
    )


def test_draws_at_both_ends_of_the_range_pick_kept_tokens() -> None:
    # Tokens 1 and 3 survive top-k. A draw of exactly 0 must land on the most likely token
    # and the largest value below 1 on the last kept one, not on a dropped token next to it.
    logits = torch.tensor([0.1, 0.4, 0.2, 0.3]).log().expand(2, -1)
    uniform = torch.tensor([0.0, math.nextafter(1.0, 0.0)])
    tokens = sample(logits, batch(*[SamplingParams(top_k=2)] * 2), uniform)
    assert tokens.tolist() == [1, 3]


def test_a_draw_above_the_rounded_total_picks_a_kept_token() -> None:
    # Over a large vocabulary the running sum of probabilities often ends a few bits below 1,
    # so the largest draw below 1 can exceed every entry of it.
    logits = 4 * torch.randn(8, 50_000, generator=torch.Generator().manual_seed(1))
    params = batch(*[SamplingParams(top_k=1000)] * 8)
    tokens = sample(logits, params, torch.full((8,), math.nextafter(1.0, 0.0)))
    assert torch.all(token_probs(logits, params).gather(-1, tokens.unsqueeze(-1)) > 0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_logits_are_sampled_in_fp32(dtype: torch.dtype) -> None:
    logits = torch.randn(4, 1000, generator=torch.Generator().manual_seed(6)).to(dtype)
    params = batch(*[SamplingParams(temperature=0.7, top_p=0.9)] * 4)
    probs = token_probs(logits, params)
    assert probs.dtype == torch.float32
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(4))
    tokens = sample(logits, params, torch.rand(4, generator=torch.Generator().manual_seed(7)))
    assert torch.all(probs.gather(-1, tokens.unsqueeze(-1)) > 0)


def test_from_params_builds_one_entry_per_row() -> None:
    greedy, sampled = SamplingParams(temperature=0), SamplingParams(0.8, top_k=40, top_p=0.9)
    mixed = batch(greedy, sampled)
    assert mixed.temperature.tolist() == pytest.approx([0.0, 0.8])
    assert mixed.top_k.tolist() == [0, 40]
    assert mixed.top_p.tolist() == pytest.approx([1.0, 0.9])
    assert mixed.min_p.tolist() == [0.0, 0.0]
    assert not mixed.all_greedy
    assert batch(greedy, greedy).all_greedy


def test_from_params_flags_batches_with_penalties() -> None:
    plain = SamplingParams()
    assert not batch(plain, plain).has_penalties
    for penalized in (
        SamplingParams(repetition_penalty=1.2),
        SamplingParams(presence_penalty=0.5),
        SamplingParams(frequency_penalty=-0.5),
    ):
        assert batch(plain, penalized).has_penalties
    mixed = batch(plain, SamplingParams(repetition_penalty=1.2, frequency_penalty=0.5))
    assert mixed.repetition_penalty.tolist() == pytest.approx([1.0, 1.2])
    assert mixed.presence_penalty.tolist() == [0.0, 0.0]
    assert mixed.frequency_penalty.tolist() == [0.0, 0.5]


def test_token_counts_count_every_occurrence() -> None:
    counts = token_counts([[1, 3, 1, 1], [], [0, 4]], vocab_size=5, device=CPU)
    assert counts.tolist() == [[0, 3, 0, 1, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]]
    assert token_counts([], vocab_size=5, device=CPU).shape == (0, 5)


def penalized(
    logits: list[float], params: SamplingParams, prompt: list[int], output: list[int]
) -> Tensor:
    """Applies one row's penalties and returns its logits [V]."""
    return apply_penalties(
        torch.tensor([logits]),
        batch(params),
        token_counts([prompt], len(logits), CPU),
        token_counts([output], len(logits), CPU),
    )[0]


def test_repetition_penalty_lowers_tokens_from_the_prompt_and_the_output() -> None:
    # Token 0 is in the prompt, tokens 1 and 3 in the output, token 2 is new.
    logits = penalized(
        [2.0, 4.0, 3.0, -1.0], SamplingParams(repetition_penalty=2.0), prompt=[0], output=[1, 3, 3]
    )
    # Positive logits are halved. The negative one is doubled, which also lowers it, and
    # occurring twice does not penalize it twice.
    assert logits.tolist() == [1.0, 2.0, 3.0, -2.0]


def test_presence_penalty_is_subtracted_once_per_generated_token() -> None:
    logits = penalized(
        [2.0, 4.0, 3.0, -1.0], SamplingParams(presence_penalty=0.5), prompt=[0], output=[1, 3, 3]
    )
    assert logits.tolist() == [2.0, 3.5, 3.0, -1.5]


def test_frequency_penalty_grows_with_each_occurrence() -> None:
    logits = penalized(
        [2.0, 4.0, 3.0, -1.0], SamplingParams(frequency_penalty=0.5), prompt=[0], output=[1, 3, 3]
    )
    assert logits.tolist() == [2.0, 3.5, 3.0, -2.0]


def test_negative_penalties_encourage_repetition() -> None:
    params = SamplingParams(presence_penalty=-1.0, frequency_penalty=-0.5)
    logits = penalized([2.0, 4.0, 3.0, -1.0], params, prompt=[], output=[1, 3, 3])
    assert logits.tolist() == [2.0, 5.5, 3.0, 1.0]


def test_default_penalties_leave_logits_unchanged() -> None:
    logits = torch.randn(1, 30, generator=torch.Generator().manual_seed(8))
    counts = token_counts([[1, 2, 2, 7]], 30, CPU)
    assert torch.equal(apply_penalties(logits, batch(SamplingParams()), counts, counts), logits)


def test_rows_keep_their_own_penalties() -> None:
    logits = torch.tensor([[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]])
    params = batch(
        SamplingParams(),
        SamplingParams(repetition_penalty=2.0),
        SamplingParams(frequency_penalty=1.0),
    )
    output_counts = token_counts([[1, 1], [1, 1], [1, 1]], 2, CPU)
    result = apply_penalties(logits, params, torch.zeros(3, 2), output_counts)
    assert result.tolist() == [[2.0, 4.0], [2.0, 2.0], [2.0, 2.0]]


def test_penalties_take_effect_before_sampling() -> None:
    # Token 0 leads until its two occurrences cost it 0.5 each.
    logits = torch.tensor([[3.0, 2.5, 0.0]])
    params = batch(SamplingParams(temperature=0, frequency_penalty=0.5))
    counts = token_counts([[0, 0]], 3, CPU)
    assert sample(logits, params, torch.rand(1)).tolist() == [0]
    penalized_logits = apply_penalties(logits, params, torch.zeros(1, 3), counts)
    assert sample(penalized_logits, params, torch.rand(1)).tolist() == [1]


def test_half_precision_logits_are_penalized_in_fp32() -> None:
    logits = torch.tensor([[2.0, 4.0, -1.0]], dtype=torch.bfloat16)
    counts = token_counts([[0, 2]], 3, CPU)
    result = apply_penalties(logits, batch(SamplingParams(repetition_penalty=1.5)), counts, counts)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.tensor([[2.0 / 1.5, 4.0, -1.5]]))


def seeded(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_draws_lie_in_the_unit_interval() -> None:
    uniform = draw_uniform([None] * 1000 + [seeded(0)] * 1000, shared=seeded(1))
    assert uniform.shape == (2000,)
    assert uniform.dtype == torch.float32
    assert torch.all((uniform >= 0) & (uniform < 1))


def test_seeded_rows_draw_the_same_numbers_in_any_batch() -> None:
    # The same seed is used alone, then in the last row of a larger batch whose other rows
    # draw from a different shared generator.
    steps = 5
    generator = seeded(42)
    alone = torch.stack([draw_uniform([generator], shared=seeded(0))[0] for _ in range(steps)])
    generator, shared = seeded(42), seeded(7)
    in_batch = torch.stack(
        [draw_uniform([None, seeded(step), None, generator], shared)[3] for step in range(steps)]
    )
    assert torch.equal(alone, in_batch)
    assert len(set(alone.tolist())) == steps


def test_rows_without_a_seed_draw_from_the_shared_generator() -> None:
    first = draw_uniform([None, seeded(42), None], shared=seeded(0))
    second = draw_uniform([None, seeded(42), None], shared=seeded(1))
    assert first[1] == second[1]
    assert first[0] != second[0]
    assert first[2] != second[2]
    assert first[0] != first[2]
    assert torch.equal(first, draw_uniform([None, seeded(42), None], shared=seeded(0)))


def test_different_seeds_draw_different_numbers() -> None:
    uniform = draw_uniform([seeded(1), seeded(2), seeded(1)], shared=seeded(0))
    assert uniform[0] != uniform[1]
    assert uniform[0] == uniform[2]


def test_a_seed_reproduces_the_sampled_tokens() -> None:
    logits = torch.randn(6, 3, 100, generator=seeded(9))  # [steps, B, V]
    params = batch(*[SamplingParams(temperature=1.2, top_p=0.95)] * 3)

    def generate(seed: int, row: int, shared_seed: int) -> list[int]:
        generators: list[torch.Generator | None] = [None, None, None]
        generators[row] = seeded(seed)
        shared = seeded(shared_seed)
        # The same logits are placed in the seeded row at every step.
        tokens = [
            sample(step_logits[[0, 0, 0]], params, draw_uniform(generators, shared))[row]
            for step_logits in logits
        ]
        return torch.stack(tokens).tolist()

    assert generate(seed=5, row=0, shared_seed=1) == generate(seed=5, row=2, shared_seed=2)
    assert generate(seed=5, row=0, shared_seed=1) != generate(seed=6, row=0, shared_seed=1)
