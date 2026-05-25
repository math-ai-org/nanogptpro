# Last-sum dataset

This directory contains the dedicated `last_sum` variant only.

Each example has:

- a mixed prefix of numbers and exactly one `R` token
- query token `?last`
- `=`
- the sum of integers strictly after the lone `R` token, which is the final marker

Example:

`4815 R 5839 5598 5835 ?last = 17272`

The prefix is every token before `?last`; in this example, it is
`4815 R 5839 5598 5835`, and the answer sums the numbers after the
lone `R` token.

## Generate data

From repo root:

```bash
.venv/bin/python data/last-sum/prepare.py \
  --digit_width=4 \
  --digit_width_mode=fixed \
  --min_sequence_tokens=5 \
  --max_sequence_tokens=8 \
  --num_train=1000000 \
  --num_val=10000 \
  --overwrite
```

By default, this writes generated files to:

- `data/last-sum/`

Set `NANOGPT_DATA_ROOT` to write and train from scratch instead:

- `export NANOGPT_DATA_ROOT=/scratch/gpfs/ARORA/$USER/nanogptpro_mod/data`

You can override a single run with:

- `--output_dir=/custom/path`

Important: this `prepare.py` intentionally requires those generation knobs to be specified
explicitly. If you omit them, it exits with a reminder instead of silently relying on defaults.

Written files follow the usual row-aligned arithmetic format:

- `train_ids_last_sum_fixed_d4.bin`
- `val_ids_last_sum_fixed_d4.bin`
- `train_context_mask_last_sum_fixed_d4.bin`
- `val_context_mask_last_sum_fixed_d4.bin`
- `meta_last_sum_fixed_d4.pkl`

Optional held-out eval length generation:

```bash
.venv/bin/python data/last-sum/prepare.py \
  --digit_width=4 \
  --digit_width_mode=fixed \
  --min_sequence_tokens=1 \
  --max_sequence_tokens=32 \
  --eval_sequence_tokens=48 \
  --num_train=1000000 \
  --num_val=10000 \
  --overwrite
```

With `--eval_sequence_tokens`, the base train/val splits are still emitted at the configured
`min_sequence_tokens..max_sequence_tokens` range, and additional eval files are written with
length-specific suffixes such as:

- `val_ids_last_sum_fixed_d4_eval@l48.bin`
- `val_context_mask_last_sum_fixed_d4_eval@l48.bin`
- `meta_last_sum_fixed_d4_eval@l48.pkl`

Multiple lengths are supported via CSV (deduplicated and sorted), for example
`--eval_sequence_tokens=33,40,48`.
Each eval length must be greater than `--max_sequence_tokens` so the split is actually held out.

## Train

Example launch:

```bash
.venv/bin/python -m nanogptpro.train_adam_arithmetics \
  config/last_sum/train_gpt_mha_rope_small_adam_last_sum_l16_d4_single_gpu.py
```

The length-specific configs expect matching dataset directories such as `last-sum-l16`,
`last-sum-l32`, or `last-sum-l64`. Prepare those by passing an explicit output
directory, for example `--output_dir=data/last-sum-l16`.

Ordinary train/val runs use online generation by default. Online last-sum uses
the same task text, tokenizer metadata, padding, and contextual loss mask as
this `prepare.py`; length-style dataset names such as `last-sum-l16` infer
fixed sequence length, and names like `last-sum-vd5to32` infer the `5..32`
range. Pass `--data_source=prepared` to read prepared `.bin` artifacts or
held-out `eval_sequence_tokens` splits.
