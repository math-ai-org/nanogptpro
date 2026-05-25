# Arithmetics dataset: fixed-digit addition (byte-tokenized)

This dataset is for contextual pretraining on fixed-width addition in the form:

`[aaaa] + [bbbb] = [ccccc]`

- `a` and `b` are fixed-width `x` digits (leading zeros allowed).
- `c` is fixed-width `x+1` digits (leading zeros allowed) to accommodate carry.
- **Reverse encoding** is used per number: `1234` is written as `4321`.
- The training objective is **contextual**:
  - Context: everything up to and including `=` (i.e. `[...] + [...] =`)
  - Text: everything after `=` (i.e. the space + `[ccccc]`)
  - Tokens in the context portion do **not** contribute to loss.

Tokenization is switchable:

- `tokenizer=byteoss` (default): byte tokenizer (0..255) + GPT-OSS special tokens (199998..200018).
  - Each example is `<|startoftext|>...<|return|>` and is written as `uint32`.
  - Training scripts default to `byteoss_vocab=compact`, remapping special token IDs so the model uses
    `vocab_size=277` (smaller embedding/LM head). Use `--byteoss_vocab=sparse` to keep `vocab_size=200019`.
- `tokenizer=gpt2`: `tiktoken` GPT-2 BPE.
  - Each example is wrapped with `<|beginoftext|>`...`<|endoftext|>` and padded to a fixed `seq_len`
    with `<|padding|>` (written as `uint16`).

## Generate data

From repo root:

`python data/arithmetics-addition-fixed_digit/prepare.py --digit_width=4 --num_train=1000000 --num_val=10000`

To switch tokenizer:

`python data/arithmetics-addition-fixed_digit/prepare.py --tokenizer=gpt2 --encoding=gpt2 --digit_width=4`

Note: `--encoding` is ignored when using the default `--tokenizer=byteoss`.

By default, generated outputs are written to:

- `data/arithmetics-addition-fixed_digit/`

Set `NANOGPT_DATA_ROOT` to write and train from scratch instead:

- `export NANOGPT_DATA_ROOT=/scratch/gpfs/ARORA/$USER/nanogptpro_mod/data`

You can override a single run with:

- `--output_dir=/custom/path`

Written files are
namespaced by `digit_width` so multiple variants can coexist (e.g. `*_d4.*`):

- `train_ids_d{digit_width}.bin` / `val_ids_d{digit_width}.bin` (`uint16` or `uint32` tokens; row-major `[num_examples, seq_len]`)
- `train_context_mask_d{digit_width}.bin` / `val_context_mask_d{digit_width}.bin` (uint8; 1 for context tokens, 0 for text tokens)
- `meta_d{digit_width}.pkl` (shapes + tokenizer metadata)

## Train

This dataset is stored as a 2D `[num_examples, seq_len]` token matrix, and training should be
**example-aligned** (each row is a complete supervised sample).

`nanogptpro.train_adam_arithmetics` defaults to online generation with
`--data_loader=stream`, which emits full examples without cutting across
samples (important if you expect train loss to reach ~0).

To read this prepared `.bin` dataset, pass `--data_source=prepared`.
To sample prepared examples randomly with replacement, also pass
`--data_loader=random_example`.

Online generation uses the same fixed-digit format, tokenizer metadata,
padding, and contextual loss mask as `prepare.py`, but creates examples at
batch time.
