import torch
from torch import Tensor

from kiyal.cache.contiguous import ContiguousKVCache
from kiyal.model.llama import LlamaModel


def assert_greedy_equivalent(actual: Tensor, expected: Tensor, atol: float) -> None:
    """Checks that an optimized computation gives the reference's logits within `atol`.

    An optimization changes the shapes and the order of floating-point operations, so its
    logits differ from the reference in the last bits and cannot be compared exactly.

    Agreement within `atol` also settles the greedy tokens. The top token can lose at most
    `atol` and any other token can gain at most `atol`, so the top token stays on top
    wherever it leads by more than 2 * atol. Greedy decoding can differ only at the
    remaining steps, which `near_tie_steps` finds.

    Args:
        actual: [steps, V] logits of the optimized computation.
        expected: [steps, V] logits of the reference computation, for the same inputs.
        atol: Largest difference allowed between the two.
    """
    assert actual.shape == expected.shape
    max_diff = (actual - expected).abs().max().item()
    assert max_diff <= atol, f"logits differ by {max_diff:.3g}, more than {atol:.3g}"


def near_tie_steps(logits: Tensor, atol: float) -> Tensor:
    """Returns the steps at which a difference within `atol` could change the greedy token.

    Args:
        logits: [steps, V] logits of the reference computation.
        atol: Largest difference allowed between two computations of the logits.

    Returns:
        [steps] booleans, True where the top two logits are within 2 * atol of each other.
    """
    top2 = logits.topk(2, dim=-1).values  # [steps, 2]
    return (top2[:, 0] - top2[:, 1]) <= 2 * atol


@torch.inference_mode()
def single_greedy(model: LlamaModel, prompt: list[int], num_steps: int) -> tuple[Tensor, Tensor]:
    """Decodes one prompt alone with a KV cache, the reference for the batching rungs.

    Returns:
        The [num_steps, V] logits of every step and the [num_steps] tokens chosen from them.
    """
    cache = ContiguousKVCache(model.config, 1, len(prompt) + num_steps - 1)
    positions = torch.arange(len(prompt)).unsqueeze(0)  # [1, T]
    logits = [model(torch.tensor([prompt]), positions, cache=cache)]  # prefill, [1, V]
    tokens = [logits[-1].argmax(dim=-1)]  # [1] each
    for _ in range(num_steps - 1):
        position = torch.tensor([[cache.length]])  # [1, 1]
        logits.append(model(tokens[-1].view(1, 1), position, cache=cache))  # decode
        tokens.append(logits[-1].argmax(dim=-1))
    return torch.cat(logits), torch.cat(tokens)
