from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Final, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nanogptpro.utils.prepare_output_paths import resolve_output_dir
from nanogptpro.utils.dataset_publish import clear_metadata_invalidation_marker, invalidate_metadata_file
from nanogptpro.utils.nanogpt_tokenizers import (
    TiktokenEncodingLike,
    TokenizerName,
    get_tiktoken_encoding,
    numpy_token_dtype,
    tokenizer_meta,
)

_SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
_ASCII_ZERO: Final[int] = ord("0")


def _digit_suffix(*, digit_width: int) -> str:
    return f"d{digit_width}"


class ArithmeticsAdditionVariableDigitPreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokenizer: TokenizerName = Field(default="byteoss", description="Tokenizer: gpt2|byteoss.")
    encoding: str = Field(
        default="gpt2",
        description="tiktoken encoding name (ignored using default byteoss tokenizer).",
    )

    digit_width: PositiveInt = Field(
        default=4,
        description="Maximum number of digits per addend (x). Each sample uses widths in [1, x].",
    )
    num_train: PositiveInt = Field(default=1_000_000, description="Number of training examples.")
    num_val: PositiveInt = Field(default=10_000, description="Number of validation examples.")
    seed: int = Field(default=42, description="RNG seed.")

    output_dir: Path | None = Field(
        default=None,
        description=(
            "Output directory (default: NANOGPT_DATA_ROOT/arithmetics-addition-variable_digit "
            "or data/arithmetics-addition-variable_digit)."
        ),
    )
    overwrite: bool = Field(default=False, description="Overwrite existing outputs.")

    def resolved_output_dir(self) -> Path:
        return resolve_output_dir(output_dir=self.output_dir, dataset_label=_SCRIPT_DIR.name)


def _encode_ascii_bytes(s: str) -> list[int]:
    try:
        return list(s.encode("ascii"))
    except UnicodeEncodeError as exc:
        raise ValueError(f"Non-ASCII character in arithmetics text: {s!r}") from exc


def _reverse_digits(value: int) -> str:
    return str(value)[::-1]


def _sample_reversed_digits(*, rng: np.random.Generator, digit_width: int, allow_leading_zeros: bool) -> str:
    """
    Sample a decimal integer and return its digit string in *reversed* order.

    This avoids generating large Python/NumPy integers (e.g. digit_width=32) by sampling digits
    directly. For digit_width>1 and allow_leading_zeros=False, the most-significant digit is
    constrained to 1..9 (which corresponds to the final character in the reversed string).
    """
    if digit_width < 1:
        raise ValueError(f"digit_width must be >= 1, got {digit_width}")

    if digit_width == 1:
        d = int(rng.integers(0, 10))
        return chr(_ASCII_ZERO + d)

    prefix_len = digit_width - 1
    prefix = rng.integers(0, 10, size=(prefix_len,), dtype=np.uint8)
    if allow_leading_zeros:
        last = int(rng.integers(0, 10))
    else:
        last = int(rng.integers(1, 10))
    raw = (prefix + _ASCII_ZERO).tobytes() + bytes((_ASCII_ZERO + last,))
    return raw.decode("ascii")


def _add_reversed_decimal(a_rev: str, b_rev: str) -> str:
    """Return reversed digit string for int(a)+int(b), given reversed digit strings."""
    if not a_rev:
        raise ValueError("a_rev must be non-empty")
    if not b_rev:
        raise ValueError("b_rev must be non-empty")

    carry = 0
    out_chars: list[str] = []
    max_len = max(len(a_rev), len(b_rev))
    for i in range(max_len):
        da = ord(a_rev[i]) - _ASCII_ZERO if i < len(a_rev) else 0
        db = ord(b_rev[i]) - _ASCII_ZERO if i < len(b_rev) else 0
        total = da + db + carry
        out_chars.append(chr(_ASCII_ZERO + (total % 10)))
        carry = total // 10

    if carry:
        out_chars.append(chr(_ASCII_ZERO + carry))
    return "".join(out_chars)


def _make_example(*, rng: np.random.Generator, digit_width: int) -> tuple[str, str]:
    if digit_width < 1:
        raise ValueError(f"digit_width must be >= 1, got {digit_width}")

    a_width = int(rng.integers(1, digit_width + 1))
    b_width = int(rng.integers(1, digit_width + 1))

    a_rev = _sample_reversed_digits(rng=rng, digit_width=a_width, allow_leading_zeros=(a_width == 1))
    b_rev = _sample_reversed_digits(rng=rng, digit_width=b_width, allow_leading_zeros=(b_width == 1))
    c_rev = _add_reversed_decimal(a_rev, b_rev)

    context = f"[{a_rev}] + [{b_rev}] ="
    text = f" [{c_rev}]"
    return context, text


def _max_seq_len(*, digit_width: int) -> int:
    if digit_width < 1:
        raise ValueError(f"digit_width must be >= 1, got {digit_width}")
    a_rev = "9" * digit_width
    b_rev = "9" * digit_width
    c_rev = "9" * (digit_width + 1)
    context = f"[{a_rev}] + [{b_rev}] ="
    text = f" [{c_rev}]"
    return len(_encode_ascii_bytes(f"{context}{text}")) + 2  # BOS/EOS


def _encode_example_byteoss(*, context: str, text: str, bos_id: int, eos_id: int) -> list[int]:
    ids: list[int] = [bos_id]
    ids.extend(_encode_ascii_bytes(f"{context}{text}"))
    ids.append(eos_id)
    return ids


def _encode_example_gpt2(
    *,
    context: str,
    text: str,
    enc: TiktokenEncodingLike,
    bos_id: int,
    eos_id: int,
) -> tuple[list[int], int]:
    context_ids = enc.encode_ordinary(context)
    text_ids = enc.encode_ordinary(text)
    ids = [bos_id]
    ids.extend(context_ids)
    ids.extend(text_ids)
    ids.append(eos_id)
    context_len = 1 + len(context_ids)
    return ids, context_len


def _ensure_writable(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (pass --overwrite to replace).")
    tmp_path = Path(f"{path}.tmp")
    if tmp_path.exists() and not overwrite:
        raise FileExistsError(f"Found leftover temp file: {tmp_path} (delete it first or pass --overwrite).")


def _split_paths(
    *,
    output_dir: Path,
    split: Literal["train", "val"],
    digit_width: int,
) -> tuple[Path, Path]:
    digit_suffix = _digit_suffix(digit_width=digit_width)
    return (
        output_dir / f"{split}_ids_{digit_suffix}.bin",
        output_dir / f"{split}_context_mask_{digit_suffix}.bin",
    )


def _preflight_outputs(*, output_dir: Path, digit_width: int, overwrite: bool) -> None:
    splits: tuple[Literal["train", "val"], ...] = ("train", "val")
    for split in splits:
        ids_path, mask_path = _split_paths(output_dir=output_dir, split=split, digit_width=digit_width)
        _ensure_writable(ids_path, overwrite=overwrite)
        _ensure_writable(mask_path, overwrite=overwrite)

    digit_suffix = _digit_suffix(digit_width=digit_width)
    _ensure_writable(output_dir / f"meta_{digit_suffix}.pkl", overwrite=overwrite)


def _write_split(
    *,
    split: Literal["train", "val"],
    tokenizer: TokenizerName,
    enc: TiktokenEncodingLike | None,
    num_examples: int,
    digit_width: int,
    seq_len: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    token_dtype: np.dtype,
    rng: np.random.Generator,
    output_dir: Path,
    overwrite: bool,
) -> None:
    ids_path, mask_path = _split_paths(output_dir=output_dir, split=split, digit_width=digit_width)
    _ensure_writable(ids_path, overwrite=overwrite)
    _ensure_writable(mask_path, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    ids_tmp = Path(f"{ids_path}.tmp")
    mask_tmp = Path(f"{mask_path}.tmp")

    ids_mm: np.memmap | None = None
    mask_mm: np.memmap | None = None
    try:
        ids_mm = np.memmap(ids_tmp, dtype=token_dtype, mode="w+", shape=(num_examples, seq_len))
        mask_mm = np.memmap(mask_tmp, dtype=np.uint8, mode="w+", shape=(num_examples, seq_len))

        for i in range(num_examples):
            context, text = _make_example(rng=rng, digit_width=digit_width)
            if tokenizer == "byteoss":
                ids_unpadded = _encode_example_byteoss(context=context, text=text, bos_id=bos_id, eos_id=eos_id)
                actual_len = len(ids_unpadded)
                if actual_len > seq_len:
                    raise RuntimeError(
                        f"Unexpected token_seq_len={actual_len} (max {seq_len}) for example: {context + text!r}"
                    )
                ids = ids_unpadded + [pad_id] * (seq_len - actual_len)
                context_len = 1 + len(_encode_ascii_bytes(context))
            elif tokenizer == "gpt2":
                if enc is None:
                    raise RuntimeError("Missing tiktoken Encoding for tokenizer='gpt2'.")
                ids_unpadded, context_len = _encode_example_gpt2(
                    context=context,
                    text=text,
                    enc=enc,
                    bos_id=bos_id,
                    eos_id=eos_id,
                )
                actual_len = len(ids_unpadded)
                if actual_len > seq_len:
                    raise RuntimeError(
                        f"Unexpected token_seq_len={actual_len} (max {seq_len}) for example: {context + text!r}"
                    )
                ids = ids_unpadded + [pad_id] * (seq_len - actual_len)
            else:
                raise ValueError(f"Unsupported tokenizer={tokenizer!r}")

            mask = [1] * seq_len
            for j in range(context_len, actual_len):
                mask[j] = 0

            ids_mm[i, :] = np.asarray(ids, dtype=token_dtype)
            mask_mm[i, :] = np.asarray(mask, dtype=np.uint8)

        ids_mm.flush()
        mask_mm.flush()
    except Exception:
        ids_mm = None
        mask_mm = None
        try:
            ids_tmp.unlink()
        except OSError:
            pass
        try:
            mask_tmp.unlink()
        except OSError:
            pass
        raise
    else:
        ids_mm = None
        mask_mm = None
        ids_tmp.replace(ids_path)
        mask_tmp.replace(mask_path)


def _parse_config(argv: Sequence[str] | None = None) -> ArithmeticsAdditionVariableDigitPreprocessConfig:
    parser = argparse.ArgumentParser(
        description="Arithmetics preprocessing (variable-width addition, tokenizer switchable)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="byteoss",
        choices=["gpt2", "byteoss"],
        help="Tokenizer name (default: byteoss)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="gpt2",
        help="tiktoken encoding name (ignored using default byteoss tokenizer; default: gpt2)",
    )
    parser.add_argument(
        "--digit_width",
        type=int,
        default=4,
        help="Maximum addend digit width x (samples use widths in [1, x]; default: 4)",
    )
    parser.add_argument("--num_train", type=int, default=1_000_000, help="Train examples (default: 1000000)")
    parser.add_argument("--num_val", type=int, default=10_000, help="Val examples (default: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Output dir (default: NANOGPT_DATA_ROOT/arithmetics-addition-variable_digit "
            "or data/arithmetics-addition-variable_digit)"
        ),
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="Overwrite outputs")
    args = parser.parse_args(argv)

    try:
        return ArithmeticsAdditionVariableDigitPreprocessConfig.model_validate(vars(args))
    except ValidationError as e:
        raise SystemExit(str(e)) from e


def main(argv: Sequence[str] | None = None) -> None:
    cfg = _parse_config(argv)

    output_dir = cfg.resolved_output_dir()
    digit_width = int(cfg.digit_width)
    digit_suffix = _digit_suffix(digit_width=digit_width)
    meta = tokenizer_meta(cfg.tokenizer, encoding=cfg.encoding)
    token_dtype = numpy_token_dtype(meta.token_dtype)

    bos_id = int(meta.bos_token_id)
    eos_id = int(meta.eos_token_id)
    pad_id = int(meta.pad_token_id)

    enc = get_tiktoken_encoding(cfg.encoding) if cfg.tokenizer == "gpt2" else None
    seq_len = _max_seq_len(digit_width=digit_width)

    _preflight_outputs(output_dir=output_dir, digit_width=digit_width, overwrite=bool(cfg.overwrite))
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / f"meta_{digit_suffix}.pkl"
    invalidate_metadata_file(meta_path)

    rng = np.random.default_rng(int(cfg.seed))
    _write_split(
        split="train",
        tokenizer=cfg.tokenizer,
        enc=enc,
        num_examples=int(cfg.num_train),
        digit_width=digit_width,
        seq_len=seq_len,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        token_dtype=token_dtype,
        rng=rng,
        output_dir=output_dir,
        overwrite=bool(cfg.overwrite),
    )
    _write_split(
        split="val",
        tokenizer=cfg.tokenizer,
        enc=enc,
        num_examples=int(cfg.num_val),
        digit_width=digit_width,
        seq_len=seq_len,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        token_dtype=token_dtype,
        rng=rng,
        output_dir=output_dir,
        overwrite=bool(cfg.overwrite),
    )

    _ensure_writable(meta_path, overwrite=bool(cfg.overwrite))
    meta_tmp = Path(f"{meta_path}.tmp")
    stop_token_ids = [eos_id]
    special_tokens = meta.special_tokens or {}
    if "<|call|>" in special_tokens:
        stop_token_ids.append(int(special_tokens["<|call|>"]))

    payload = meta.model_dump() | {
        "eot_token_id": eos_id,
        "stop_token_ids": stop_token_ids,
        "digit_width": digit_width,
        "digit_width_min": 1,
        "digit_width_max": digit_width,
        "digit_width_mode": "variable",
        "seq_len": seq_len,
        "block_size": seq_len - 1,
    }
    meta_tmp.write_bytes(pickle.dumps(payload))
    meta_tmp.replace(meta_path)
    clear_metadata_invalidation_marker(meta_path)

    example_context, example_text = _make_example(rng=np.random.default_rng(0), digit_width=digit_width)
    preview = f"{meta.bos_token}{example_context}{example_text}{meta.eos_token}"
    print(f"wrote arithmetics-addition-variable_digit dataset to {output_dir}")
    print(f"tokenizer={cfg.tokenizer} vocab_size={meta.vocab_size} seq_len={seq_len} block_size={seq_len - 1}")
    print(f"example: {preview!r} (context={example_context!r} text={example_text!r})")


if __name__ == "__main__":
    main()
