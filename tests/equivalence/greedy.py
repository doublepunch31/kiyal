from torch import Tensor


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
