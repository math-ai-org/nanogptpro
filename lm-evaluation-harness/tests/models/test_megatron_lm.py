from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from lm_eval.models.megatron_lm import _clone_layer_spec_with_attention_mask


def test_clone_layer_spec_with_attention_mask_does_not_mutate_original() -> None:
    """Attention-mask overrides are scoped to cloned Megatron spec components."""
    self_attention = SimpleNamespace(params={"other": "value"})
    submodules = SimpleNamespace(self_attention=self_attention)
    layer_spec = SimpleNamespace(submodules=submodules)

    updated_layer_spec, did_update = _clone_layer_spec_with_attention_mask(
        layer_spec,
        "arbitrary",
    )
    updated = cast("Any", updated_layer_spec)

    assert did_update is True
    assert updated_layer_spec is not layer_spec
    assert updated.submodules is not submodules
    assert updated.submodules.self_attention is not self_attention
    assert self_attention.params == {"other": "value"}
    assert updated.submodules.self_attention.params == {
        "other": "value",
        "attn_mask_type": "arbitrary",
    }


def test_clone_layer_spec_with_attention_mask_skips_uncopyable_attention() -> None:
    """Uncopyable spec components are left untouched instead of being mutated."""

    class UncopyableAttention:
        def __init__(self) -> None:
            self.params = {"other": "value"}

        def __copy__(self) -> UncopyableAttention:
            return self

    self_attention = UncopyableAttention()
    submodules = SimpleNamespace(self_attention=self_attention)
    layer_spec = SimpleNamespace(submodules=submodules)

    updated_layer_spec, did_update = _clone_layer_spec_with_attention_mask(
        layer_spec,
        "arbitrary",
    )

    assert did_update is False
    assert updated_layer_spec is layer_spec
    assert self_attention.params == {"other": "value"}
