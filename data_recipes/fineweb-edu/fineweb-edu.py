"""
FineWeb-EDU dataset preprocessing (for SRS pretraining).
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

example doc to highlight the structure of the dataset:
{
  "text": "Posted by mattsmith on 20th April 2012\nStraight from...",
  "id": "<urn:uuid:d853d453-196e-4488-a411-efc2b26c40d2>",
  "dump": "CC-MAIN-2013-20",
  "url": "http://nleastchatter.com/philliesphandom/tag/freddy-galvis/",
  "date": "2013-05-18T07:24:47Z",
  "file_path": "s3://commoncrawl/long.../path.../file.gz",
  "language": "en",
  "language_score": 0.9185474514961243,
  "token_count": 594
}
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
from itertools import islice
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from datasets import load_dataset
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, model_validator

from nanogptpro.data.nanogpt_tokenizers import (
    DEFAULT_GPT_OSS_TOKENIZER_ID,
    TokenizerName,
    numpy_token_dtype,
    resolve_distinct_padding_token_id,
    tokenizer_meta,
)
from nanogptpro.utils.dataset_publish import clear_metadata_invalidation_marker, invalidate_metadata_file
from nanogptpro.utils.prepare_output_paths import default_data_root
from nanogptpro.utils.text_shard_preprocess import (
    DATAFILE_VERSION,
    HEADER_BYTES,
    HEADER_INTS,
    MAGIC,
    init_text_tokenizer,
    tokenize_text_record,
    write_text_shards,
)
from nanogptpro.utils.text_shards import remove_stale_text_shard_artifacts

FinewebEduVersion = Literal["10B", "100B", "350B"]
SUPPORTED_VERSIONS: tuple[FinewebEduVersion, ...] = ("10B", "100B", "350B")

_PRESETS: dict[FinewebEduVersion, tuple[str, str]] = {
    "10B": ("fineweb-edu10B", "sample-10BT"),
    "100B": ("fineweb-edu100B", "sample-100BT"),
    "350B": ("fineweb-edu350B", "sample-350BT"),
}

_SCRIPT_DIR = Path(__file__).resolve().parent


class FinewebEduPreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: FinewebEduVersion = "100B"
    shard_size: PositiveInt = Field(default=10**8, description="Size of each shard in tokens.")
    num_procs: PositiveInt | None = Field(default=None, description="Multiprocessing workers (default: auto).")
    chunksize: PositiveInt = Field(default=16, description="Multiprocessing Pool.imap() chunksize.")
    val_shards: PositiveInt = Field(default=1, description="Number of initial shards reserved for validation.")
    streaming: bool = Field(default=False, description="Use Hugging Face streaming mode (no local download).")
    hf_cache_dir: Path | None = Field(default=None, description="Hugging Face datasets cache dir (optional).")
    output_dir: Path | None = Field(default=None, description="Output directory (defaults to version preset).")
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
    max_docs: PositiveInt | None = Field(default=None, description="Stop after N documents (debugging).")

    # Aligned streaming/packing ("Fit or Cut") during `.bin` writing.
    fit_or_cut: bool = Field(
        default=False,
        description="Enable Fit-or-Cut packing to avoid short orphan fragments at the start of blocks.",
    )
    block_size: PositiveInt = Field(
        default=4096,
        description="Model context window used for Fit-or-Cut packing (must match training `block_size`).",
    )
    fit_or_cut_threshold: PositiveInt = Field(
        default=100,
        description="Minimum viable fragment length for wrap padding/truncation decisions.",
    )

    @model_validator(mode="after")
    def _validate_fit_or_cut(self) -> FinewebEduPreprocessConfig:
        if not self.fit_or_cut:
            return self
        block_size = int(self.block_size)
        threshold = int(self.fit_or_cut_threshold)
        if threshold >= block_size:
            raise ValueError(
                f"fit_or_cut_threshold must be < block_size, got threshold={threshold} block_size={block_size}"
            )
        return self

    def preset(self) -> tuple[str, str]:
        return _PRESETS[self.version]

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        local_dir, _ = self.preset()
        return default_data_root() / _SCRIPT_DIR.name / local_dir


def _default_num_procs() -> int:
    cpu_count = os.cpu_count() or 1
    return min(32, max(1, cpu_count - 2))


def tokenize(doc: Mapping[str, Any]) -> np.ndarray:
    return tokenize_text_record(doc, text_column="text")


def _parse_config(argv: Sequence[str] | None = None) -> FinewebEduPreprocessConfig:
    parser = argparse.ArgumentParser(description="FineWeb-EDU dataset preprocessing")
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="100B",
        choices=SUPPORTED_VERSIONS,
        help="Which version to use: 10B|100B|350B",
    )
    parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
    parser.add_argument("--num_procs", type=int, default=None, help="Multiprocessing workers (default: auto)")
    parser.add_argument("--chunksize", type=int, default=16, help="Multiprocessing Pool.imap() chunksize")
    parser.add_argument("--val_shards", type=int, default=1, help="Number of initial shards reserved for validation")
    parser.add_argument("--streaming", action="store_true", help="Stream from HF instead of downloading")
    parser.add_argument("--hf_cache_dir", type=Path, default=None, help="HF datasets cache directory (optional)")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory (optional)")
    parser.add_argument(
        "--tokenizer", type=str, default="gpt2", choices=["gpt2", "gpt-oss", "byteoss"], help="Tokenizer name"
    )
    parser.add_argument("--encoding", type=str, default="gpt2", help="tiktoken encoding name")
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
    parser.add_argument("--max_docs", type=int, default=None, help="Stop after N documents (debugging)")
    parser.add_argument(
        "--fit_or_cut",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Fit-or-Cut packing (default: False). Use --fit_or_cut to enable.",
    )
    parser.add_argument("--block_size", type=int, default=4096, help="Model context window for Fit-or-Cut packing.")
    parser.add_argument("--fit_or_cut_threshold", type=int, default=100, help="Minimum viable fragment length.")

    ns = parser.parse_args(argv)
    try:
        return FinewebEduPreprocessConfig.model_validate(vars(ns))
    except ValidationError as exc:
        parser.error(str(exc))
        raise  # unreachable, but keeps type checkers happy


def main() -> None:
    config = _parse_config()
    output_dir = config.resolved_output_dir()
    local_dir, remote_name = config.preset()
    nprocs = int(config.num_procs or _default_num_procs())
    meta = tokenizer_meta(config.tokenizer, encoding=config.encoding, hf_tokenizer_id=config.hf_tokenizer_id)
    token_dtype = numpy_token_dtype(meta.token_dtype)
    padding_token_id = resolve_distinct_padding_token_id(
        vocab_size=int(meta.vocab_size),
        bos_token_id=int(meta.bos_token_id),
        eos_token_id=int(meta.eos_token_id),
        pad_token_id=int(meta.pad_token_id),
    )

    print(
        "FineWeb-EDU preprocessing config:\n"
        f"- version: {config.version} (HF name={remote_name}, local_dir={local_dir})\n"
        f"- output_dir: {output_dir}\n"
        f"- shard_size: {config.shard_size:,} tokens\n"
        f"- val_shards: {config.val_shards}\n"
        f"- streaming: {config.streaming}\n"
        f"- num_procs: {nprocs}\n"
        f"- chunksize: {config.chunksize}\n"
        f"- tokenizer: {config.tokenizer}\n"
        f"- encoding: {config.encoding!r}\n"
        f"- hf_tokenizer_id: {config.hf_tokenizer_id!r}\n"
        f"- allow_special_tokens: {config.allow_special_tokens}\n"
        f"- max_docs: {config.max_docs}\n"
        f"- fit_or_cut: {config.fit_or_cut}\n"
        f"- block_size: {config.block_size}\n"
        f"- fit_or_cut_threshold: {config.fit_or_cut_threshold}\n"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "meta.pkl"

    load_kwargs: dict[str, Any] = {"name": remote_name, "split": "train", "streaming": config.streaming}
    if config.hf_cache_dir is not None:
        load_kwargs["cache_dir"] = str(config.hf_cache_dir)
    fw = load_dataset("HuggingFaceFW/fineweb-edu", **load_kwargs)

    docs_iter: Any = fw
    if config.max_docs is not None:
        docs_iter = islice(docs_iter, int(config.max_docs))

    init_text_tokenizer(config.tokenizer, config.encoding, config.hf_tokenizer_id, bool(config.allow_special_tokens))
    invalidate_metadata_file(meta_path)
    with mp.Pool(
        nprocs,
        initializer=init_text_tokenizer,
        initargs=(config.tokenizer, config.encoding, config.hf_tokenizer_id, bool(config.allow_special_tokens)),
    ) as pool:
        token_stream = pool.imap(tokenize, docs_iter, chunksize=int(config.chunksize))
        shard_layout = write_text_shards(
            token_stream,
            output_dir=output_dir,
            shard_size=int(config.shard_size),
            val_shards=int(config.val_shards),
            token_dtype=token_dtype,
            fit_or_cut=bool(config.fit_or_cut),
            block_size=int(config.block_size),
            fit_or_cut_threshold=int(config.fit_or_cut_threshold),
            padding_token_id=padding_token_id,
        )
    remove_stale_text_shard_artifacts(output_dir, layout=shard_layout)

    payload = meta.model_dump() | {
        "tokenization": {"allow_special_tokens": bool(config.allow_special_tokens)},
        "datafile": {
            "magic": MAGIC,
            "version": DATAFILE_VERSION,
            "header_ints": HEADER_INTS,
            "header_bytes": HEADER_BYTES,
        },
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
        "shards": shard_layout.model_dump(),
    }
    meta_tmp = Path(f"{meta_path}.tmp")
    meta_tmp.write_bytes(pickle.dumps(payload))
    os.replace(meta_tmp, meta_path)
    clear_metadata_invalidation_marker(meta_path)


if __name__ == "__main__":
    mp.freeze_support()
    main()
