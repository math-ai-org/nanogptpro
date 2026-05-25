"""NanoGPT Pro model bridge for lm-evaluation-harness.

This repository vendors EleutherAI's lm-evaluation-harness under `lm-evaluation-harness/`.
The harness expects HuggingFace-compatible `PreTrainedModel` and `PretrainedConfig` classes.

NanoGPT Pro provides GPT variants under the `nanogptpro.model` package
(e.g. `gpt-mha-rope`, `gpt-mha-alibi`, `gpt-gqa-rope`). This module exposes
`GPTConfig` and `GPT` symbols that delegate to the appropriate NanoGPT Pro
implementation, selected by:

1) `NANOGPTPRO_MODEL_TYPE` environment variable, or
2) `nanogptpro_model_type` stored in the saved `config.json`, or
3) heuristics based on the checkpoint path (e.g. `output/out_<model_type>_.../checkpoint-...`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Protocol, cast

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


class _GPTModule(Protocol):
    GPTConfig: type[PretrainedConfig]
    GPT: type[PreTrainedModel]


eval_logger = logging.getLogger(__name__)
_RUN_DIR_MODEL_TYPE_RE = re.compile(r"^out_(?P<model_type>.+?)_\d+m_Opt_")


def _looks_like_nanogptpro_repo_root(candidate: Path) -> bool:
    """Heuristic for finding a NanoGPT Pro checkout root."""
    if not (candidate / "nanogptpro" / "model").is_dir():
        return False
    pyproject = candidate / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        contents = pyproject.read_text(encoding="utf-8")
    except OSError:
        return True
    return 'name = "nanogptpro"' in contents


def _maybe_infer_model_type_from_path(pretrained_model_name_or_path: str) -> str | None:
    """Infer `model_type` from NanoGPT Pro training output directories.

    Expected layout:
        .../output/out_<model_type>_<Nm>_Opt_<...>/checkpoint-<step>
    """
    path = Path(pretrained_model_name_or_path)
    for candidate in (path, *path.parents):
        match = _RUN_DIR_MODEL_TYPE_RE.match(candidate.name)
        if match is not None:
            model_type = match.group("model_type")
            if model_type:
                return model_type
    return None


def _ensure_nanogptpro_on_sys_path(
    *, pretrained_model_name_or_path: str | None = None
) -> None:
    repo_root_env = os.environ.get("NANOGPTPRO_REPO_ROOT")
    candidates: list[Path] = []
    if repo_root_env:
        candidates.append(Path(repo_root_env).expanduser().resolve())

    # new_model.py -> models/ -> lm_eval/ -> lm-evaluation-harness/ -> repo root
    try:
        candidates.append(Path(__file__).resolve().parents[3])
    except IndexError:
        pass

    try:
        cwd = Path.cwd().resolve()
    except OSError:
        cwd = None

    if cwd is not None:
        candidates.append(cwd)
        candidates.extend(list(cwd.parents))

    if pretrained_model_name_or_path is not None:
        try:
            checkpoint_path = Path(pretrained_model_name_or_path).expanduser().resolve()
        except OSError:
            checkpoint_path = Path(pretrained_model_name_or_path)
        candidates.append(checkpoint_path)
        candidates.extend(list(checkpoint_path.parents))

    repo_root: Path | None = None
    for candidate in candidates:
        if _looks_like_nanogptpro_repo_root(candidate):
            repo_root = candidate
            break
    if repo_root is None:
        for candidate in candidates:
            if (candidate / "nanogptpro" / "model").is_dir():
                repo_root = candidate
                break
    if repo_root is None:
        return

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _resolve_model_type(
    config_dict: dict[str, Any], *, pretrained_model_name_or_path: str | None = None
) -> str:
    explicit = os.environ.get("NANOGPTPRO_MODEL_TYPE")
    if explicit:
        return explicit

    stored = config_dict.get("nanogptpro_model_type")
    if isinstance(stored, str) and stored:
        return stored

    if pretrained_model_name_or_path is not None:
        inferred_from_path = _maybe_infer_model_type_from_path(
            pretrained_model_name_or_path
        )
        if inferred_from_path is not None:
            return inferred_from_path

    raise ValueError(
        "Unable to determine NanoGPT Pro model module name. "
        "Set NANOGPTPRO_MODEL_TYPE, or ensure the checkpoint config.json contains "
        "`nanogptpro_model_type` (recommended)."
    )


def _import_nanogpt_module(
    model_type: str, *, pretrained_model_name_or_path: str | None = None
) -> _GPTModule:
    _ensure_nanogptpro_on_sys_path(
        pretrained_model_name_or_path=pretrained_model_name_or_path
    )
    try:
        from nanogptpro.model_registry import load_model_module

        return cast("_GPTModule", load_model_module(model_type))
    except (ModuleNotFoundError, ValueError) as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            f"Unable to import NanoGPT Pro model module for model_id={model_type!r}. "
            "Set NANOGPTPRO_REPO_ROOT to the nanogptpro repo root and "
            "NANOGPTPRO_MODEL_TYPE to a valid model module name."
        ) from exc


def _get_checkpoint_commit_mismatch_warning(
    pretrained_model_name_or_path: str,
) -> str | None:
    checkpoint_dir = Path(pretrained_model_name_or_path)
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not trainer_state_path.is_file():
        return None
    try:
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(trainer_state, dict):
        return None
    try:
        from nanogptpro.utils.git_utils import (
            get_checkpoint_commit_mismatch_warning,
            get_repo_git_commit,
        )
    except (ImportError, OSError, RuntimeError):
        return None
    return get_checkpoint_commit_mismatch_warning(
        current_commit=get_repo_git_commit(),
        checkpoint_commit=trainer_state.get("codebase_git_commit"),
        checkpoint_dir=checkpoint_dir,
    )


class GPTConfig(PretrainedConfig):
    model_type = "nanogpt-pro"

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any
    ) -> PretrainedConfig:
        config_dict, kwargs = PretrainedConfig.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        model_type = _resolve_model_type(
            config_dict, pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        module = _import_nanogpt_module(
            model_type, pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        return module.GPTConfig.from_dict(config_dict, **kwargs)


class GPT(PreTrainedModel):
    config_class = GPTConfig

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any
    ) -> PreTrainedModel:
        commit_mismatch_warning = _get_checkpoint_commit_mismatch_warning(
            pretrained_model_name_or_path
        )
        if commit_mismatch_warning is not None:
            eval_logger.warning(commit_mismatch_warning)
        config = kwargs.get("config")
        if config is not None:
            config_dict = config.to_dict()
        else:
            config_dict, _ = PretrainedConfig.get_config_dict(
                pretrained_model_name_or_path, **kwargs
            )
        model_type = _resolve_model_type(
            config_dict, pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        module = _import_nanogpt_module(
            model_type, pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        return module.GPT.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
