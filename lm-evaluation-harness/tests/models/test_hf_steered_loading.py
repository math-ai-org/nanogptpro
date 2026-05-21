from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch


def test_steered_model_loads_torch_config_with_weights_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from lm_eval.models import hf_steered

    captured_load_kwargs: dict[str, bool] = {}

    class _FakeModel:
        dtype: torch.dtype = torch.float32

    def _fake_hflm_init(self: Any, *args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        self._device = torch.device("cpu")
        self._model = _FakeModel()

    def _fake_torch_load(
        file_obj: object,
        *,
        weights_only: bool,
    ) -> dict[str, dict[str, object]]:
        _ = file_obj
        captured_load_kwargs["weights_only"] = weights_only
        return {
            "layers.0": {
                "action": "add",
                "steering_vector": torch.ones(2),
                "steering_coefficient": 2.0,
                "bias": None,
                "head_index": None,
            }
        }

    monkeypatch.setattr(hf_steered.HFLM, "__init__", _fake_hflm_init)
    monkeypatch.setattr(hf_steered.torch, "load", _fake_torch_load)
    steer_path = tmp_path / "steer.pt"
    steer_path.write_bytes(b"unused by patched torch.load")

    model = hf_steered.SteeredModel(
        pretrained="unused",
        steer_path=str(steer_path),
    )

    assert captured_load_kwargs == {"weights_only": True}
    assert list(model.hook_to_steer) == ["layers.0"]
