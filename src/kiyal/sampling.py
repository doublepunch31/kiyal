from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from kiyal.engine.sequence import SamplingParams


@dataclass(frozen=True, slots=True)
class SamplingTensors:
    """Sampling settings of a batch as tensors with one entry per row.

    Rows keep their own settings, so one batch can mix greedy and sampled requests.
    """

    temperature: Tensor  # [B]
    top_k: Tensor  # [B], 0 where the filter is off
    top_p: Tensor  # [B]
    min_p: Tensor  # [B]
    repetition_penalty: Tensor  # [B]
    presence_penalty: Tensor  # [B]
    frequency_penalty: Tensor  # [B]
    # Known on the host, so skipping the sort for a greedy batch, or the token counts for a
    # batch without penalties, needs no read from the device.
    all_greedy: bool
    has_penalties: bool

    @classmethod
    def from_params(cls, params: Sequence[SamplingParams], device: torch.device) -> Self:
        return cls(
            temperature=torch.tensor([p.temperature for p in params], device=device),
            top_k=torch.tensor([p.top_k for p in params], device=device),
            top_p=torch.tensor([p.top_p for p in params], device=device),
            min_p=torch.tensor([p.min_p for p in params], device=device),
            repetition_penalty=torch.tensor([p.repetition_penalty for p in params], device=device),
            presence_penalty=torch.tensor([p.presence_penalty for p in params], device=device),
            frequency_penalty=torch.tensor([p.frequency_penalty for p in params], device=device),
            all_greedy=all(p.temperature == 0 for p in params),
            has_penalties=any(
                p.repetition_penalty != 1 or p.presence_penalty != 0 or p.frequency_penalty != 0
                for p in params
            ),
        )


def draw_uniform(generators: Sequence[torch.Generator | None], shared: torch.Generator) -> Tensor:
    """Draws one number from [0, 1) for each row of a batch.

    A row with its own generator draws from it, so a seeded request gets the same numbers
    whichever other requests share its batch. The remaining rows draw from `shared`.

    Generators stay on the CPU: a seed then gives the same draws on every device, and one
    number per row is cheap to copy over.

    Args:
        generators: The generator of each row, or None for rows without a seed.
        shared: CPU generator for the rows without their own.

    Returns:
        [B] draws, on the CPU.
    """
    uniform = torch.rand(len(generators), generator=shared)  # [B]
    for row, generator in enumerate(generators):
        if generator is not None:
            uniform[row] = torch.rand((), generator=generator)
    return uniform


def token_counts(
    token_ids: Sequence[Sequence[int]], vocab_size: int, device: torch.device
) -> Tensor:
    """Counts how often each token occurs in each row.

    Args:
        token_ids: The tokens of each row. Rows may differ in length.
        vocab_size: Number of token ids, V.
        device: Device of the result.

    Returns:
        [B, V] counts in fp32.
    """
    rows = [row for row, ids in enumerate(token_ids) for _ in ids]
    columns = [token for ids in token_ids for token in ids]
    counts = torch.zeros((len(token_ids), vocab_size), device=device)  # [B, V]
    index = (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(columns, dtype=torch.long, device=device),
    )
    # accumulate adds 1 for every occurrence instead of writing 1 once per distinct token.
    return counts.index_put_(index, torch.ones(len(columns), device=device), accumulate=True)


def apply_penalties(
    logits: Tensor, params: SamplingTensors, prompt_counts: Tensor, output_counts: Tensor
) -> Tensor:
    """Lowers the logits of tokens a row has already used.

    The repetition penalty covers the prompt and the output. The presence and frequency
    penalties count generated tokens only, as in the OpenAI API, so that a prompt does not
    penalize the words needed to answer it.

    Args:
        logits: [B, V] next-token logits.
        params: Settings of the B rows.
        prompt_counts: [B, V] occurrences of each token in the row's prompt.
        output_counts: [B, V] occurrences of each token in the row's output so far.

    Returns:
        [B, V] penalized logits in fp32.
    """
    logits = logits.float()  # [B, V]
    # Repetition penalty from CTRL (Keskar et al., 2019). Dividing a negative logit would
    # raise it, so negative logits are multiplied instead.
    penalty = params.repetition_penalty.unsqueeze(-1)  # [B, 1]
    seen = (prompt_counts + output_counts) > 0  # [B, V]
    logits = torch.where(seen, torch.where(logits > 0, logits / penalty, logits * penalty), logits)

    generated = output_counts > 0  # [B, V]
    logits = logits - params.presence_penalty.unsqueeze(-1) * generated
    return logits - params.frequency_penalty.unsqueeze(-1) * output_counts


def sample(logits: Tensor, params: SamplingTensors, uniform: Tensor) -> Tensor:
    """Picks the next token of each row.

    Sampling is a deterministic function of its inputs: all randomness comes in through
    `uniform`, so the caller decides which generator each row draws from.

    Args:
        logits: [B, V] next-token logits.
        params: Settings of the B rows.
        uniform: [B] numbers drawn uniformly from [0, 1). Greedy rows ignore theirs.

    Returns:
        [B] token ids.
    """
    greedy_tokens = logits.argmax(dim=-1)  # [B]
    if params.all_greedy:
        return greedy_tokens

    probs, token_ids = _sorted_token_probs(logits, params)  # [B, V] each
    cdf = probs.cumsum(dim=-1)  # [B, V]
    # Inverse transform sampling: the token at rank i owns the interval (cdf[i-1], cdf[i]],
    # whose length is its probability, and a draw picks the first rank whose cdf reaches it.
    rank = torch.searchsorted(cdf, uniform.to(cdf).unsqueeze(-1))  # [B, 1]
    # Dropped tokens sit after the kept ones and own empty intervals. Rounding can leave the
    # cdf's last value just below 1, and a draw above it would land past the kept tokens.
    num_kept = (probs > 0).sum(dim=-1, keepdim=True)  # [B, 1]
    rank = torch.minimum(rank, num_kept - 1)
    sampled_tokens = token_ids.gather(-1, rank).squeeze(-1)  # [B]
    return torch.where(params.temperature == 0, greedy_tokens, sampled_tokens)


def token_probs(logits: Tensor, params: SamplingTensors) -> Tensor:
    """Returns the distribution each row samples its next token from.

    Temperature, top-k, top-p and min-p are applied in that order. A row with temperature 0
    puts all of its probability on the most likely token.

    Args:
        logits: [B, V] next-token logits.
        params: Settings of the B rows.

    Returns:
        [B, V] probabilities in fp32. Each row sums to 1.
    """
    sorted_probs, token_ids = _sorted_token_probs(logits, params)  # [B, V] each
    probs = torch.zeros_like(sorted_probs).scatter_(-1, token_ids, sorted_probs)  # [B, V]
    one_hot = torch.zeros_like(probs).scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
    return torch.where((params.temperature == 0).unsqueeze(-1), one_hot, probs)


def _sorted_token_probs(logits: Tensor, params: SamplingTensors) -> tuple[Tensor, Tensor]:
    """Applies temperature and the filters to tokens sorted from most to least likely.

    Returns:
        [B, V] probabilities in descending order, zero for dropped tokens, and the [B, V]
        token id at each rank. Greedy rows are computed as if their temperature were 1.
    """
    # Half precision loses small probabilities in a softmax and a running sum over a large
    # vocabulary.
    logits = logits.float()  # [B, V]
    vocab_size = logits.shape[-1]
    # A greedy row's result is not used; dividing by 1 keeps it finite.
    temperature = torch.where(params.temperature == 0, 1.0, params.temperature)  # [B]
    # One sort serves all three filters. A stable sort leaves equal logits in token-id order,
    # so ties are broken the same way on every device.
    sorted_logits, token_ids = (logits / temperature.unsqueeze(-1)).sort(
        dim=-1, descending=True, stable=True
    )  # [B, V] each

    # Top-k keeps ranks below k.
    rank = torch.arange(vocab_size, device=logits.device)  # [V]
    top_k = torch.where(params.top_k > 0, params.top_k, vocab_size).unsqueeze(-1)  # [B, 1]
    keep = rank < top_k  # [B, V]
    probs = torch.softmax(sorted_logits.masked_fill(~keep, float("-inf")), dim=-1)  # [B, V]

    # Top-p keeps a token while the probability ranked above it is below top_p. That is the
    # smallest set reaching top_p, and it always holds the most likely token. top_p = 1 is
    # checked directly because rounding in the running sum could drop tokens at the tail.
    top_p = params.top_p.unsqueeze(-1)  # [B, 1]
    mass_above = probs.cumsum(dim=-1) - probs  # [B, V]
    keep &= (mass_above < top_p) | (top_p >= 1)

    # Min-p compares each token with the most likely one. Renormalizing after top-p scales
    # both by the same factor, so the probabilities from before top-p give the same result.
    keep &= probs >= params.min_p.unsqueeze(-1) * probs[:, :1]

    probs = probs * keep
    return probs / probs.sum(dim=-1, keepdim=True), token_ids
