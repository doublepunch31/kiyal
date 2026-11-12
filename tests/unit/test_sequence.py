from typing import Any

import pytest

from kiyal.engine.sequence import SamplingParams


def test_defaults_sample_from_the_unfiltered_distribution() -> None:
    params = SamplingParams()
    assert (params.temperature, params.top_k, params.top_p, params.min_p) == (1.0, 0, 1.0, 0.0)
    penalties = (params.repetition_penalty, params.presence_penalty, params.frequency_penalty)
    assert penalties == (1.0, 0.0, 0.0)
    assert params.seed is None


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
    ],
)
def test_values_out_of_range_raise(settings: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SamplingParams(**settings)
