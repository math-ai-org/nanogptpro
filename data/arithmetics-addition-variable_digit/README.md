# Arithmetics dataset: variable-digit addition (byte-tokenized)

This dataset is for contextual pretraining on **variable-width** addition in the form:

`[a...] + [b...] = [c...]`

- Each addend uses a random digit width in `[1, x]` (where `x = digit_width` in `prepare.py`).
- Addends are sampled uniformly by digit width (then uniformly within that width).
  - Width `1`: `0..9`
  - Width `w>1`: `10^(w-1)..10^w-1` (no leading zeros)
- The sum `c = a + b` is written with its natural width (`1..x+1` digits).
- **Reverse encoding** is used per number: `1234` is written as `4321`.
- The training objective is **contextual**:
  - Context: everything up to and including `=` (i.e. `[...] + [...] =`)
  - Text: everything after `=` (i.e. the space + `[c...]`)
  - Tokens in the context portion do **not** contribute to loss.

Tokenization is switchable:

- `tokenizer=byteoss` (default): byte tokenizer (0..255) + GPT-OSS special tokens (199998..200018).
  - Each example is `<|startoftext|>...<|return|>` and is written as `uint32`.
  - Examples are padded to a fixed `seq_len` using `<|endoftext|>` so all rows have shape
    `[num_examples, seq_len]`. Padding tokens are masked out of loss.
- `tokenizer=gpt2`: `tiktoken` GPT-2 BPE.
  - Each example is wrapped with `<|beginoftext|>`...`<|endoftext|>` and padded to a fixed `seq_len`
    with `<|padding|>` (written as `uint16`).

## Generate data

From repo root:

`python data/arithmetics-addition-variable_digit/prepare.py --digit_width=4 --num_train=1000000 --num_val=10000`

To switch tokenizer:

`python data/arithmetics-addition-variable_digit/prepare.py --tokenizer=gpt2 --encoding=gpt2 --digit_width=4`

By default, generated outputs are written to:

- `data/arithmetics-addition-variable_digit/`

Set `NANOGPT_DATA_ROOT` to write and train from scratch instead:

- `export NANOGPT_DATA_ROOT=/scratch/gpfs/ARORA/$USER/nanogptpro_mod/data`

You can override a single run with:

- `--output_dir=/custom/path`

Written files are
namespaced by `digit_width` so multiple variants can coexist (e.g. `*_d4.*`):

- `train_ids_d{digit_width}.bin` / `val_ids_d{digit_width}.bin` (`uint16` or `uint32` tokens; row-major `[num_examples, seq_len]`)
- `train_context_mask_d{digit_width}.bin` / `val_context_mask_d{digit_width}.bin` (uint8; 1 for context/pad tokens, 0 for text tokens)
- `meta_d{digit_width}.pkl` (shapes + tokenizer metadata)

## Train

This dataset is stored as a 2D `[num_examples, seq_len]` token matrix, and training should be
**example-aligned** (each row is a complete supervised sample).

`nanogptpro.train_adam_arithmetics` defaults to online generation with
`--data_loader=stream`, which emits full examples without cutting across
samples.

To read this prepared `.bin` dataset, pass `--data_source=prepared`.
Online generation uses the same variable-digit format, tokenizer metadata,
padding, and contextual loss mask as `prepare.py`, but creates examples at
batch time.
