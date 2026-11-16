import pytest
import torch

from tests.equivalence.greedy import assert_greedy_equivalent, near_tie_steps

ATOL = 0.01
# Top-2 gaps of 1.0, 0.03 and 0.015: only the last is within 2 * ATOL.
EXPECTED = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.03, 1.0], [1.0, 0.0, 1.015]])


def test_near_ties_are_the_steps_with_a_gap_within_twice_the_tolerance() -> None:
    assert near_tie_steps(EXPECTED, ATOL).tolist() == [False, False, True]


def test_equal_logits_are_equivalent() -> None:
    assert_greedy_equivalent(EXPECTED.clone(), EXPECTED, ATOL)


def test_a_changed_token_at_a_near_tie_is_allowed() -> None:
    actual = EXPECTED.clone()
    actual[2] = torch.tensor([1.009, 0.0, 1.006])
    assert actual[2].argmax() != EXPECTED[2].argmax()
    assert_greedy_equivalent(actual, EXPECTED, ATOL)


def test_a_changed_token_without_a_near_tie_fails() -> None:
    actual = EXPECTED.clone()
    actual[1] = torch.tensor([0.0, 1.01, 1.02])
    assert actual[1].argmax() != EXPECTED[1].argmax()
    with pytest.raises(AssertionError, match="logits differ"):
        assert_greedy_equivalent(actual, EXPECTED, ATOL)


def test_a_difference_above_the_tolerance_fails_even_with_the_same_tokens() -> None:
    actual = EXPECTED.clone()
    actual[0, 2] += 0.02
    assert actual.argmax(dim=-1).tolist() == EXPECTED.argmax(dim=-1).tolist()
    with pytest.raises(AssertionError, match=r"logits differ by 0\.02, more than 0\.01"):
        assert_greedy_equivalent(actual, EXPECTED, ATOL)


def test_different_shapes_fail() -> None:
    with pytest.raises(AssertionError):
        assert_greedy_equivalent(EXPECTED[:2], EXPECTED, ATOL)


def test_logits_within_the_tolerance_keep_the_token_where_there_is_no_near_tie() -> None:
    generator = torch.Generator().manual_seed(0)
    expected = torch.randn(2000, 8, generator=generator) * 0.05
    noise = (torch.rand(expected.shape, generator=generator) * 2 - 1) * ATOL
    actual = expected + noise
    assert_greedy_equivalent(actual, expected, ATOL * 1.0001)

    same_token = actual.argmax(dim=-1) == expected.argmax(dim=-1)
    near_tie = near_tie_steps(expected, ATOL * 1.0001)
    assert same_token[~near_tie].all()
    # The input must contain both kinds of step, and tokens that did change at near ties.
    assert 0 < near_tie.sum() < len(near_tie)
    assert not same_token[near_tie].all()
