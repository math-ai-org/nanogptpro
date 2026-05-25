from __future__ import annotations

import logging
from random import Random
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Any, TypeVar

    _T = TypeVar("_T")

eval_logger = logging.getLogger(__name__)


class ContextSampler:
    def __init__(
        self,
        df: Sequence[dict[str, Any]] | None = None,
        *,
        rnd: int | Random | None = None,
        fewshot_indices: list[int] | None = None,
        **kwargs,
    ) -> None:
        self.rnd = rnd if isinstance(rnd, Random) else Random(rnd)
        self.df = df or []
        self.fewshot_indices = fewshot_indices
        self._loaded = False  # to iterate over fewshot_indices when needed

    def sample(
        self,
        n: int,
        eval_doc: dict[str, Any] | None = None,
        df: Sequence[dict[str, Any]] | None = None,
        **kwargs,
    ) -> Sequence[dict[str, Any]]:
        """
        Sample n documents from the pool.

        Args:
            n: Number of documents to sample
            eval_doc: Optional document to exclude from sampling
            df: Optional list of documents to sample from

        Returns:
            List of sampled documents
        """
        assert n >= 0, "Error: number of samples requested must be >=0"
        if n == 0:
            return []

        if df:
            self.df = df

        docs = self.fewshot_docs()
        doc_count = len(docs)
        assert doc_count > 0, "Error: no documents available for sampling."

        if eval_doc is None:
            available_count = doc_count
            assert available_count >= n, (
                f"Error: number of fewshot samples requested ({n}) exceeds the "
                f"{available_count} that are available."
            )
            sampled_indices = self.rnd.sample(range(doc_count), n)
        else:
            sampled_indices = self._sample_indices_excluding_eval_doc(docs, n, eval_doc)

        res = [docs[idx] for idx in sampled_indices]
        assert len(res) == n, (
            f"Error: number of fewshot samples returned ({len(res)}) not equal to number requested ({n})."
        )
        return res

    def _sample_indices_excluding_eval_doc(
        self,
        docs: Sequence[dict[str, Any]],
        n: int,
        eval_doc: dict[str, Any],
    ) -> list[int]:
        doc_count = len(docs)
        if n * 2 >= doc_count:
            available_indices = self._available_indices_excluding_eval_doc(
                docs, eval_doc
            )
            available_count = len(available_indices)
            assert available_count >= n, (
                f"Error: number of fewshot samples requested ({n}) exceeds the "
                f"{available_count} that are available."
            )
            return self.rnd.sample(available_indices, n)

        selected_indices: list[int] = []
        seen_indices: set[int] = set()
        max_attempts = min(doc_count, max(n * 8, 32))

        for _ in range(max_attempts):
            if len(selected_indices) == n:
                return selected_indices
            idx = self.rnd.randrange(doc_count)
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            if docs[idx] == eval_doc:
                continue
            selected_indices.append(idx)

        remaining_indices = [
            idx
            for idx in range(doc_count)
            if idx not in seen_indices and docs[idx] != eval_doc
        ]
        available_count = len(selected_indices) + len(remaining_indices)
        assert available_count >= n, (
            f"Error: number of fewshot samples requested ({n}) exceeds the "
            f"{available_count} that are available."
        )
        selected_indices.extend(
            self.rnd.sample(remaining_indices, n - len(selected_indices))
        )
        return selected_indices

    @staticmethod
    def _available_indices_excluding_eval_doc(
        docs: Sequence[dict[str, Any]],
        eval_doc: dict[str, Any],
    ) -> list[int]:
        return [idx for idx, doc in enumerate(docs) if doc != eval_doc]

    def set_rnd(self, rnd: int | Random | None):
        self.rnd = rnd if isinstance(rnd, Random) else Random(rnd)
        return self

    def replace_df(self, df: Sequence[dict[str, Any]]):
        self.df = df
        self._loaded = False
        return self

    def fewshot_docs(self) -> Sequence[dict[str, Any]]:
        """Return cached fewshot docs if available"""
        if self._loaded:
            return self.df
        if self.fewshot_indices and self.df and not self._loaded:
            self.df = [self.df[i] for i in self.fewshot_indices]
        self._loaded = True
        return self.df

    @staticmethod
    def rm_eval_doc(doc: _T, _iter: Iterable[_T], n=None) -> Sequence[_T]:
        return (
            [x for x in _iter if x != doc]
            if n is None
            else [x for x in _iter if x != doc][:n]
        )


class FirstNSampler(ContextSampler):
    def sample(self, n: int, eval_doc=None, df=None, **kwargs):
        """
        Draw the first `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        assert n <= len(self.df), (
            f"Error: number of fewshot samples requested exceeds the {len(self.df)} that are available."
        )
        return self.df[:n]


class BalancedSampler(ContextSampler):
    def sample(self, n: int, eval_doc=None, df=None, **kwargs):
        """
        TODO: this should return approximately class-balanced samples from our fewshot examples.
        TODO: what order should they be in? maybe random?
        """

        raise NotImplementedError


class ManualSampler(ContextSampler):
    def sample(self, n: int, eval_doc=None, df=None, **kwargs):
        """Sample manually selected few-shot examples."""
        raise NotImplementedError


SAMPLER_REGISTRY: dict[str, type[ContextSampler]] = {
    "default": ContextSampler,
    "first_n": FirstNSampler,
}


def get_sampler(name: str):
    try:
        return SAMPLER_REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"Attempted to use contextsampler '{name}', but no sampling strategy for this name found! Supported model names: {', '.join(SAMPLER_REGISTRY.keys())}"
        ) from e
