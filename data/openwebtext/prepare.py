"""
OpenWebText preprocessing.

This script:
1) downloads (or reuses cached) Hugging Face `openwebtext`
2) creates a small validation split
3) tokenizes with either:
   - `tokenizer=gpt2`: `tiktoken` GPT-2 BPE (default), wrapping each document with `<|beginoftext|>` ... `<|endoftext|>`
   - `tokenizer=gpt-oss`: Hugging Face GPT-OSS tokenizer (default: `openai/gpt-oss-120b`), wrapping each document
     with BOS...EOS and encoding document text without special-token parsing by default
   - `tokenizer=byteoss`: UTF-8 byte tokenizer (raw bytes, no special-token parsing in document text by default)
     + GPT-OSS special tokens (wraps with `<|startoftext|>` ... `<|return|>`)
   - Pass `--allow_special_tokens` to parse special token strings inside document text (default: disabled).
4) writes `train.bin` and `val.bin` as a flat token stream (`uint16` or `uint32`, depending on tokenizer),
   optionally applying aligned streaming packing ("Fit or Cut") to pad short prefix slots or truncate short
   orphan fragments at the start of blocks (see `data/README.md`)
5) writes `meta.pkl` with tokenizer metadata

The `.bin` format is compatible with `nanogptpro.train_adam_openwebtext`, which memory-maps these
files based on `meta.pkl`.

Reference inspiration:
https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, model_validator
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nanogptpro.utils.nanogpt_tokenizers import (
    DEFAULT_GPT_OSS_TOKENIZER_ID,
    TiktokenEncodingLike,
    TokenizerName,
    byteoss_encode,
    byteoss_encode_ordinary,
    get_tiktoken_encoding,
    gpt2_encode,
    gpt_oss_encode,
    gpt_oss_encode_ordinary,
    numpy_token_dtype,
    resolve_distinct_padding_token_id,
    tokenizer_meta,
)
from nanogptpro.utils.document_bounds import (
    build_document_bounds_from_lengths,
    document_bounds_path,
    paired_publish_bounds_tmp_path,
    paired_publish_data_tmp_path,
    publish_data_and_document_bounds_files,
    recover_incomplete_paired_publish,
    write_document_bounds_npy,
)
from nanogptpro.utils.dataset_publish import clear_metadata_invalidation_marker, invalidate_metadata_file

MAX_UINT16 = 2**16
DEFAULT_VAL_FRACTION = 0.0005
DEFAULT_WRITE_SHARDS = 1024
DEFAULT_BLOCK_SIZE = 4096
DEFAULT_FIT_OR_CUT_THRESHOLD = 100

_SCRIPT_DIR = Path(__file__).resolve().parent


class OpenWebTextPreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(default="openwebtext", description="Hugging Face dataset ID.")
    hf_split: str = Field(default="train", description="Split name to load from HF.")
    hf_cache_dir: Path | None = Field(default=None, description="Hugging Face datasets cache dir (optional).")

    output_dir: Path | None = Field(default=None, description="Output directory (default: script directory).")
    tokenizer: TokenizerName = Field(default="gpt2", description="Tokenizer: gpt2|gpt-oss|byteoss.")
    encoding: str = Field(default="gpt2", description="tiktoken encoding name.")
    hf_tokenizer_id: str = Field(
        default=DEFAULT_GPT_OSS_TOKENIZER_ID,
        description="Hugging Face tokenizer ID (used when tokenizer='gpt-oss').",
    )
    allow_special_tokens: bool = Field(
        default=False,
        description="Allow parsing special token strings inside document text (default: False; treat as literal).",
    )

    val_fraction: float = Field(
        default=DEFAULT_VAL_FRACTION, gt=0.0, lt=1.0, description="Fraction of documents reserved for validation."
    )
    seed: int = Field(default=42, description="RNG seed for dataset split.")
    shuffle: bool = Field(default=True, description="Shuffle documents before splitting.")

    num_proc: PositiveInt | None = Field(default=None, description="Workers for tokenization `Dataset.map()`.")
    num_proc_load_dataset: PositiveInt | None = Field(default=None, description="Workers for `load_dataset()`.")
    map_batch_size: PositiveInt = Field(default=1_000, description="Batch size for tokenization `Dataset.map()`.")

    write_shards: PositiveInt = Field(
        default=DEFAULT_WRITE_SHARDS,
        description="Number of contiguous shards used during `.bin` writing (higher = smoother progress, more overhead).",
    )

    # Aligned streaming/packing ("Fit or Cut") during `.bin` writing.
    fit_or_cut: bool = Field(
        default=False,
        description="Enable Fit-or-Cut packing to avoid short orphan fragments at the start of blocks.",
    )
    block_size: PositiveInt = Field(
        default=DEFAULT_BLOCK_SIZE,
        description="Model context window used for Fit-or-Cut packing (must match training `block_size`).",
    )
    fit_or_cut_threshold: PositiveInt = Field(
        default=DEFAULT_FIT_OR_CUT_THRESHOLD,
        description="Minimum viable fragment length for wrap padding/truncation decisions.",
    )

    max_docs: PositiveInt | None = Field(default=None, description="Optional cap on total documents (debugging).")
    overwrite: bool = Field(default=False, description="Overwrite existing `train.bin`/`val.bin` outputs.")

    @model_validator(mode="after")
    def _validate_fit_or_cut(self) -> OpenWebTextPreprocessConfig:
        if not self.fit_or_cut:
            return self
        block_size = int(self.block_size)
        threshold = int(self.fit_or_cut_threshold)
        if threshold >= block_size:
            raise ValueError(
                f"fit_or_cut_threshold must be < block_size, got threshold={threshold} block_size={block_size}"
            )
        return self

    def resolved_output_dir(self) -> Path:
        return self.output_dir if self.output_dir is not None else _SCRIPT_DIR

    def resolved_num_proc(self) -> int:
        return int(self.num_proc or _default_num_procs())

    def resolved_num_proc_load_dataset(self) -> int:
        return int(self.num_proc_load_dataset or self.resolved_num_proc())


def _default_num_procs() -> int:
    cpu_count = os.cpu_count() or 1
    # Leave a little headroom by default.
    return min(32, max(1, cpu_count - 2))


@lru_cache(maxsize=None)
def _get_encoding(encoding_name: str) -> TiktokenEncodingLike:
    enc = get_tiktoken_encoding(encoding_name)

    max_token_value = getattr(enc, "max_token_value", None)
    if max_token_value is None:
        max_token_value = enc.n_vocab - 1
    if int(max_token_value) >= MAX_UINT16:
        raise ValueError(
            f"Encoding {encoding_name!r} produces tokens up to {max_token_value}, "
            f"but this script writes uint16 (< {MAX_UINT16})."
        )
    n_vocab = int(getattr(enc, "n_vocab", int(max_token_value) + 1))
    if n_vocab >= MAX_UINT16:
        raise ValueError(
            f"Encoding {encoding_name!r} has n_vocab={n_vocab}, which leaves no room "
            "for a distinct BOS token in uint16 output."
        )
    eot = enc.eot_token
    if eot is None or eot < 0 or eot >= MAX_UINT16:
        raise ValueError(f"Invalid eot_token={eot} for uint16 output.")

    return enc


def tokenize_batch(
    examples: Mapping[str, list[Any]], *, encoding: str, allow_special_tokens: bool = False
) -> dict[str, list[list[int]] | list[int]]:
    """Tokenize a batch of examples; wraps each document with BOS ... EOS."""
    enc = _get_encoding(encoding)
    meta = tokenizer_meta("gpt2", encoding=encoding)
    bos_id = int(meta.bos_token_id)
    eos_id = int(meta.eos_token_id)
    allow_special_tokens = bool(allow_special_tokens)

    texts = examples.get("text", [])
    ids_batch: list[list[int]] = []
    lens: list[int] = []

    for text in texts:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        ids: list[int] = [bos_id]
        if allow_special_tokens:
            ids.extend(gpt2_encode(text, encoding=encoding, allow_special_tokens=True))
        else:
            ids.extend(enc.encode_ordinary(text))
        ids.append(eos_id)
        ids_batch.append(ids)
        lens.append(len(ids))

    return {"ids": ids_batch, "len": lens}


def tokenize_batch_byteoss(
    examples: Mapping[str, list[Any]], *, allow_special_tokens: bool = False
) -> dict[str, list[list[int]] | list[int]]:
    """
    Tokenize a batch of examples; wraps each document with <|startoftext|> ... <|return|>.

    By default, the document text itself is encoded as raw UTF-8 bytes (no special-token parsing), so raw
    substrings like `<|startoftext|>`/`<|return|>`/`<|endoftext|>` in the source text are treated as
    literal text and do not inject special token IDs. Enable `allow_special_tokens=True` to parse known
    `<|...|>` token strings into their special token IDs.
    """
    meta = tokenizer_meta("byteoss")
    bos_id = int(meta.bos_token_id)
    eos_id = int(meta.eos_token_id)
    allow_special_tokens = bool(allow_special_tokens)

    texts = examples.get("text", [])
    ids_batch: list[list[int]] = []
    lens: list[int] = []

    for text in texts:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        ids = [bos_id]
        if allow_special_tokens:
            ids.extend(byteoss_encode(text))
        else:
            ids.extend(byteoss_encode_ordinary(text))
        ids.append(eos_id)
        ids_batch.append(ids)
        lens.append(len(ids))

    return {"ids": ids_batch, "len": lens}


def tokenize_batch_gpt_oss(
    examples: Mapping[str, list[Any]], *, hf_tokenizer_id: str, allow_special_tokens: bool = False
) -> dict[str, list[list[int]] | list[int]]:
    """
    Tokenize a batch of examples with the HF GPT-OSS tokenizer; wraps each document with BOS ... EOS.

    By default, document text is encoded without parsing added/special tokens; set `allow_special_tokens=True`
    to parse special token strings inside the document text.
    """
    meta = tokenizer_meta("gpt-oss", hf_tokenizer_id=hf_tokenizer_id)
    bos_id = int(meta.bos_token_id)
    eos_id = int(meta.eos_token_id)
    allow_special_tokens = bool(allow_special_tokens)

    texts = examples.get("text", [])
    ids_batch: list[list[int]] = []
    lens: list[int] = []

    for text in texts:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        ids: list[int] = [bos_id]
        if allow_special_tokens:
            ids.extend(gpt_oss_encode(text, hf_tokenizer_id=hf_tokenizer_id))
        else:
            ids.extend(gpt_oss_encode_ordinary(text, hf_tokenizer_id=hf_tokenizer_id))
        ids.append(eos_id)
        ids_batch.append(ids)
        lens.append(len(ids))

    return {"ids": ids_batch, "len": lens}


def _ensure_writable_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (pass --overwrite to replace).")
    tmp_path = Path(f"{path}.tmp")
    if tmp_path.exists() and not overwrite:
        raise FileExistsError(f"Found leftover temp file: {tmp_path} (delete it first or pass --overwrite).")


def _fit_or_cut_decision(
    *,
    cursor: int,
    doc_len: int,
    block_size: int,
    threshold: int,
    allow_padding: bool = False,
) -> tuple[int, int, int]:
    """
    Decide how much padding to insert and how many document tokens to keep.

    The packing "cursor" is the current fill level of the active block (0..block_size-1).
    If writing `doc_len` tokens would wrap into a new block and leave a short tail fragment
    at the start of that new block (0 < remainder < threshold), we usually drop the last
    `remainder` tokens so the document ends exactly at a block boundary. If that would keep
    only a short document prefix in the current block, we instead pad the current block and
    keep the whole document for the next block.

    Returns:
        (padding_len, kept_len, new_cursor)
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    if cursor < 0 or cursor >= block_size:
        raise ValueError(f"cursor must be in [0, block_size), got cursor={cursor} block_size={block_size}")
    if doc_len < 0:
        raise ValueError(f"doc_len must be >= 0, got {doc_len}")
    if doc_len == 0:
        return 0, 0, cursor

    total = cursor + doc_len
    remainder = total % block_size

    if remainder == 0:
        return 0, doc_len, 0

    # Only truncate when the document actually wraps into a new block. If total < block_size,
    # remainder reflects within-block cursor and "dropping remainder tokens" would be invalid.
    if total > block_size and remainder < threshold:
        kept_len = doc_len - remainder
        if kept_len <= 0:
            raise RuntimeError(
                "Fit-or-Cut truncation kept no tokens; this should be impossible when total > block_size "
                f"(cursor={cursor} doc_len={doc_len} block_size={block_size} remainder={remainder})."
            )
        prefix_len = block_size - cursor
        if allow_padding and cursor > 0 and prefix_len < threshold:
            padding_len = prefix_len
            _, kept_len_at_zero, cursor_at_zero = _fit_or_cut_decision(
                cursor=0,
                doc_len=doc_len,
                block_size=block_size,
                threshold=threshold,
                allow_padding=False,
            )
            return padding_len, kept_len_at_zero, cursor_at_zero
        return 0, kept_len, 0

    return 0, doc_len, remainder


def write_tokenized_split(dset: Dataset, path: Path, *, write_shards: int, dtype: np.dtype) -> tuple[int, int]:
    """
    Writes `dset['ids']` into a `.bin` file (flat uint16 token stream).

    Returns:
        (num_docs, num_tokens)
    """
    return write_tokenized_split_packed(
        dset,
        path,
        write_shards=write_shards,
        dtype=dtype,
        fit_or_cut=False,
        block_size=DEFAULT_BLOCK_SIZE,
        fit_or_cut_threshold=DEFAULT_FIT_OR_CUT_THRESHOLD,
    )


def write_tokenized_split_packed(
    dset: Dataset,
    path: Path,
    *,
    write_shards: int,
    dtype: np.dtype,
    fit_or_cut: bool,
    block_size: int,
    fit_or_cut_threshold: int,
    padding_token_id: int | None = None,
) -> tuple[int, int]:
    """
    Writes `dset['ids']` into a `.bin` file (flat token stream).

    When `fit_or_cut=True`, applies aligned streaming packing described in `data/README.md`:
    it pads short prefix slots and truncates short orphan tail fragments.

    Returns:
        (num_docs, num_tokens_written)
    """
    num_docs = int(len(dset))
    write_shards = int(max(1, min(write_shards, max(1, num_docs))))
    block_size_i = int(block_size)
    threshold_i = int(fit_or_cut_threshold)
    padding_token_id_i = int(padding_token_id) if padding_token_id is not None else None
    if fit_or_cut and padding_token_id_i is None:
        raise ValueError("Fit-or-Cut packing requires padding_token_id for short-prefix padding.")

    if not fit_or_cut:
        token_count = int(np.sum(dset["len"], dtype=np.uint64))
        truncated_docs = 0
        truncated_tokens = 0
        padded_docs = 0
        padding_tokens = 0
    else:
        cursor = 0
        token_count = 0
        truncated_docs = 0
        truncated_tokens = 0
        padded_docs = 0
        padding_tokens = 0
        for shard_idx in tqdm(range(write_shards), desc=f"scanning {path.name} for fit-or-cut"):
            batch = dset.shard(num_shards=write_shards, index=shard_idx, contiguous=True).with_format("numpy")
            lens = batch["len"]
            for doc_len_raw in lens:
                doc_len = int(doc_len_raw)
                padding_len, kept_len, cursor = _fit_or_cut_decision(
                    cursor=cursor,
                    doc_len=doc_len,
                    block_size=block_size_i,
                    threshold=threshold_i,
                    allow_padding=True,
                )
                token_count += padding_len + kept_len
                if padding_len > 0:
                    padded_docs += 1
                    padding_tokens += padding_len
                if kept_len != doc_len:
                    truncated_docs += 1
                    truncated_tokens += doc_len - kept_len

    if token_count <= 0:
        raise ValueError(f"Expected >0 tokens for {path}, got {token_count}.")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = paired_publish_data_tmp_path(path)
    bounds_tmp_path = paired_publish_bounds_tmp_path(path)
    if fit_or_cut and truncated_docs > 0:
        print(
            f"writing {path} ({num_docs:,} docs, {token_count:,} tokens; "
            f"fit-or-cut truncated {truncated_docs:,} docs / {truncated_tokens:,} tokens; "
            f"padded {padded_docs:,} docs / {padding_tokens:,} tokens)"
        )
    elif fit_or_cut and padded_docs > 0:
        print(
            f"writing {path} ({num_docs:,} docs, {token_count:,} tokens; "
            f"fit-or-cut padded {padded_docs:,} docs / {padding_tokens:,} tokens)"
        )
    else:
        print(f"writing {path} ({num_docs:,} docs, {token_count:,} tokens)")
    arr: np.memmap | None = None
    idx = 0
    try:
        arr = np.memmap(tmp_path, dtype=dtype, mode="w+", shape=(token_count,))

        cursor = 0
        document_bounds_spans: list[tuple[int, int]] = []
        has_padding = False
        for shard_idx in tqdm(range(write_shards), desc=f"writing {path.name}"):
            batch = dset.shard(num_shards=write_shards, index=shard_idx, contiguous=True).with_format("numpy")
            ids_list = batch["ids"]
            lens = batch["len"]
            if len(ids_list) == 0:
                continue
            if not fit_or_cut:
                arr_batch = np.concatenate(ids_list).astype(dtype, copy=False)
            else:
                kept_ids_list: list[np.ndarray[Any, Any]] = []
                batch_token_offset = 0
                for ids, doc_len_raw in zip(ids_list, lens, strict=False):
                    doc_len = int(doc_len_raw)
                    padding_len, kept_len, cursor = _fit_or_cut_decision(
                        cursor=cursor,
                        doc_len=doc_len,
                        block_size=block_size_i,
                        threshold=threshold_i,
                        allow_padding=True,
                    )
                    if padding_len > 0:
                        if padding_token_id_i is None:
                            raise RuntimeError("Cannot write Fit-or-Cut padding without padding_token_id.")
                        kept_ids_list.append(np.full((padding_len,), padding_token_id_i, dtype=dtype))
                        batch_token_offset += padding_len
                        has_padding = True
                    if kept_len <= 0:
                        continue
                    doc_start = idx + batch_token_offset
                    document_bounds_spans.append((doc_start, doc_start + kept_len))
                    if kept_len == doc_len:
                        kept_ids_list.append(np.asarray(ids))
                    else:
                        kept_ids_list.append(np.asarray(ids[:kept_len]))
                    batch_token_offset += kept_len
                if not kept_ids_list:
                    continue
                arr_batch = np.concatenate(kept_ids_list).astype(dtype, copy=False)
            arr[idx : idx + arr_batch.size] = arr_batch
            idx += int(arr_batch.size)

        if idx != token_count:
            raise RuntimeError(f"Token count mismatch for {path}: wrote {idx:,} tokens, expected {token_count:,}.")

        arr.flush()
        if fit_or_cut:
            document_bounds = np.asarray(document_bounds_spans, dtype=np.int64)
            write_document_bounds_npy(bounds_tmp_path, document_bounds, full=not has_padding)
        else:
            document_bounds = build_document_bounds_from_lengths(dset["len"])
            write_document_bounds_npy(bounds_tmp_path, document_bounds)
    except Exception:
        # Best-effort cleanup to avoid leaving a huge `.tmp` behind.
        if arr is not None:
            del arr
        try:
            tmp_path.unlink()
        except OSError:
            pass
        try:
            bounds_tmp_path.unlink()
        except OSError:
            pass
        raise
    else:
        del arr
        publish_data_and_document_bounds_files(
            path,
            data_tmp_path=tmp_path,
            bounds_tmp_path=bounds_tmp_path,
        )
    return num_docs, token_count


def _parse_config(argv: Sequence[str] | None = None) -> OpenWebTextPreprocessConfig:
    parser = argparse.ArgumentParser(description="OpenWebText preprocessing")
    parser.add_argument("--dataset_id", type=str, default="openwebtext", help="HF dataset ID (default: openwebtext)")
    parser.add_argument("--hf_split", type=str, default="train", help="HF split name to load (default: train)")
    parser.add_argument("--hf_cache_dir", type=Path, default=None, help="HF datasets cache directory (optional)")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory (default: script dir)")
    parser.add_argument(
        "--tokenizer", type=str, default="gpt2", choices=["gpt2", "gpt-oss", "byteoss"], help="Tokenizer name"
    )
    parser.add_argument("--encoding", type=str, default="gpt2", help="tiktoken encoding name (default: gpt2)")
    parser.add_argument(
        "--hf_tokenizer_id",
        type=str,
        default=DEFAULT_GPT_OSS_TOKENIZER_ID,
        help=f"Hugging Face tokenizer ID for tokenizer='gpt-oss' (default: {DEFAULT_GPT_OSS_TOKENIZER_ID}).",
    )
    parser.add_argument(
        "--allow_special_tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow parsing special token strings inside document text (default: False).",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help=f"Validation fraction (default: {DEFAULT_VAL_FRACTION})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split RNG seed (default: 42)")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle before splitting (default: True)",
    )
    parser.add_argument("--num_proc", type=int, default=None, help="Tokenization workers (default: auto)")
    parser.add_argument(
        "--num_proc_load_dataset", type=int, default=None, help="load_dataset workers (default: num_proc)"
    )
    parser.add_argument("--map_batch_size", type=int, default=1_000, help="Batch size for tokenization map()")
    parser.add_argument(
        "--write_shards", type=int, default=DEFAULT_WRITE_SHARDS, help="Contiguous shards used for writing"
    )
    parser.add_argument(
        "--fit_or_cut",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Fit-or-Cut packing (default: False). Use --fit_or_cut to enable.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"Model context window for Fit-or-Cut packing (default: {DEFAULT_BLOCK_SIZE}).",
    )
    parser.add_argument(
        "--fit_or_cut_threshold",
        type=int,
        default=DEFAULT_FIT_OR_CUT_THRESHOLD,
        help=f"Minimum viable fragment length (default: {DEFAULT_FIT_OR_CUT_THRESHOLD}).",
    )
    parser.add_argument("--max_docs", type=int, default=None, help="Optional cap on docs (debugging)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing train.bin/val.bin outputs")

    ns = parser.parse_args(argv)
    values = vars(ns)
    try:
        return OpenWebTextPreprocessConfig.model_validate(values)
    except ValidationError as exc:
        parser.error(str(exc))
        raise  # unreachable


def _load_and_split_dataset(config: OpenWebTextPreprocessConfig) -> DatasetDict:
    load_kwargs: dict[str, Any] = {
        "split": config.hf_split,
        "num_proc": config.resolved_num_proc_load_dataset(),
    }
    if config.hf_cache_dir is not None:
        load_kwargs["cache_dir"] = str(config.hf_cache_dir)

    print(f"loading HF dataset {config.dataset_id!r} split={config.hf_split!r}")
    dset: Dataset = load_dataset(config.dataset_id, **load_kwargs)
    if "text" not in dset.column_names:
        raise ValueError(f"Expected a 'text' column, got columns={dset.column_names!r}")
    if config.max_docs is not None:
        max_docs = int(min(len(dset), int(config.max_docs)))
        print(f"debug: limiting to first {max_docs:,} documents")
        dset = dset.select(range(max_docs))

    num_docs = int(len(dset))
    if num_docs < 2:
        raise ValueError(f"Need at least 2 documents to split, got {num_docs}. Increase --max_docs.")
    val_docs = int(round(float(config.val_fraction) * num_docs))
    val_docs = max(1, min(val_docs, num_docs - 1))
    split = dset.train_test_split(test_size=val_docs, seed=int(config.seed), shuffle=bool(config.shuffle))
    split["val"] = split.pop("test")
    return split


def main() -> None:
    config = _parse_config()
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    meta_path = output_dir / "meta.pkl"
    recover_incomplete_paired_publish(train_path)
    recover_incomplete_paired_publish(val_path)
    _ensure_writable_output(train_path, overwrite=bool(config.overwrite))
    _ensure_writable_output(val_path, overwrite=bool(config.overwrite))
    _ensure_writable_output(document_bounds_path(train_path), overwrite=bool(config.overwrite))
    _ensure_writable_output(document_bounds_path(val_path), overwrite=bool(config.overwrite))
    _ensure_writable_output(meta_path, overwrite=bool(config.overwrite))

    meta = tokenizer_meta(config.tokenizer, encoding=config.encoding, hf_tokenizer_id=config.hf_tokenizer_id)
    token_dtype = numpy_token_dtype(meta.token_dtype)
    padding_token_id = resolve_distinct_padding_token_id(
        vocab_size=int(meta.vocab_size),
        bos_token_id=int(meta.bos_token_id),
        eos_token_id=int(meta.eos_token_id),
        pad_token_id=int(meta.pad_token_id),
    )
    if meta.token_dtype == "uint16":
        # Validate tokenizer early (so we fail fast before downloading/tokenizing).
        _get_encoding(config.encoding)

    print(
        "OpenWebText preprocessing config:\n"
        f"- dataset_id: {config.dataset_id}\n"
        f"- hf_split: {config.hf_split}\n"
        f"- output_dir: {output_dir}\n"
        f"- tokenizer: {config.tokenizer}\n"
        f"- encoding: {config.encoding!r}\n"
        f"- hf_tokenizer_id: {config.hf_tokenizer_id!r}\n"
        f"- allow_special_tokens: {config.allow_special_tokens}\n"
        f"- val_fraction: {config.val_fraction}\n"
        f"- seed: {config.seed}\n"
        f"- shuffle: {config.shuffle}\n"
        f"- num_proc: {config.resolved_num_proc()}\n"
        f"- num_proc_load_dataset: {config.resolved_num_proc_load_dataset()}\n"
        f"- map_batch_size: {config.map_batch_size}\n"
        f"- write_shards: {config.write_shards}\n"
        f"- fit_or_cut: {config.fit_or_cut}\n"
        f"- block_size: {config.block_size}\n"
        f"- fit_or_cut_threshold: {config.fit_or_cut_threshold}\n"
        f"- max_docs: {config.max_docs}\n"
        f"- overwrite: {config.overwrite}\n"
    )

    split_dataset = _load_and_split_dataset(config)

    if config.tokenizer == "gpt2":
        tokenized = split_dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=int(config.map_batch_size),
            remove_columns=["text"],
            desc="tokenizing splits",
            num_proc=int(config.resolved_num_proc()),
            fn_kwargs={"encoding": config.encoding, "allow_special_tokens": bool(config.allow_special_tokens)},
        )
    elif config.tokenizer == "gpt-oss":
        tokenized = split_dataset.map(
            tokenize_batch_gpt_oss,
            batched=True,
            batch_size=int(config.map_batch_size),
            remove_columns=["text"],
            desc="tokenizing splits",
            num_proc=int(config.resolved_num_proc()),
            fn_kwargs={
                "hf_tokenizer_id": config.hf_tokenizer_id,
                "allow_special_tokens": bool(config.allow_special_tokens),
            },
        )
    elif config.tokenizer == "byteoss":
        tokenized = split_dataset.map(
            tokenize_batch_byteoss,
            batched=True,
            batch_size=int(config.map_batch_size),
            remove_columns=["text"],
            desc="tokenizing splits",
            num_proc=int(config.resolved_num_proc()),
            fn_kwargs={"allow_special_tokens": bool(config.allow_special_tokens)},
        )
    else:
        raise ValueError(f"Unsupported tokenizer={config.tokenizer!r}")

    invalidate_metadata_file(meta_path)
    for split_name, out_path in (("train", train_path), ("val", val_path)):
        dset = tokenized[split_name]
        write_tokenized_split_packed(
            dset,
            out_path,
            write_shards=int(config.write_shards),
            dtype=token_dtype,
            fit_or_cut=bool(config.fit_or_cut),
            block_size=int(config.block_size),
            fit_or_cut_threshold=int(config.fit_or_cut_threshold),
            padding_token_id=padding_token_id,
        )

    meta_tmp = Path(f"{meta_path}.tmp")
    meta_payload = meta.model_dump() | {
        "tokenization": {"allow_special_tokens": bool(config.allow_special_tokens)},
        "document_bounds": {
            "enabled": True,
            "format": "npy",
            "dtype": "int64",
            "suffix": ".doc_bounds.npy",
        },
        "packing": {
            "algorithm": "fit_or_cut",
            "enabled": bool(config.fit_or_cut),
            "block_size": int(config.block_size),
            "threshold": int(config.fit_or_cut_threshold),
            "padding_token_id": int(padding_token_id),
        },
    }
    meta_tmp.write_bytes(pickle.dumps(meta_payload))
    os.replace(meta_tmp, meta_path)
    clear_metadata_invalidation_marker(meta_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
