from __future__ import annotations

from math import isinf
from types import SimpleNamespace

from lm_eval.models.trtllm_causallms import TRTLLM


def _logprob(value: float, rank: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(logprob=value, rank=rank)


def test_parse_logprobs_treats_missing_actual_token_as_negative_infinity() -> None:
    outputs = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                prompt_logprobs=[
                    {1: _logprob(-0.1, 1)},
                    {99: _logprob(-0.2, 1)},
                ]
            )
        ]
    )

    continuation_logprobs, is_greedy = TRTLLM._parse_logprobs(
        [1, 2], outputs, ctxlen=1
    )

    assert isinf(continuation_logprobs) and continuation_logprobs < 0
    assert is_greedy is False
