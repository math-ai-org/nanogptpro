# Interval-sum dataset

This directory contains the dedicated `interval_sum` variant only.

Each example has:

- a mixed prefix of numbers and exactly two `R` tokens
- query token `?interval`
- `=`
- the sum of integers strictly between the two `R` tokens

Example:

`4815 R 5839 5598 R 5835 ?interval = 11437`

## Generate data

From repo root:

```bash
.venv/bin/python data/interval-sum/prepare.py \
  --digit_width=4 \
  --digit_width_mode=fixed \
  --min_sequence_tokens=5 \
  --max_sequence_tokens=8 \
  --num_train=1000000 \
  --num_val=10000 \
  --overwrite
```

By default, this writes generated files to:

- `data/interval_sum/`

Set `NANOGPT_DATA_ROOT` to write and train from scratch instead:

- `export NANOGPT_DATA_ROOT=/scratch/gpfs/ARORA/$USER/nanogptpro_mod/data`

You can override a single run with:

- `--output_dir=/custom/path`

Important: this `prepare.py` intentionally requires those generation knobs to be specified
explicitly. If you omit them, it exits with a reminder instead of silently relying on defaults.

Written files follow the usual row-aligned arithmetic format:

- `train_ids_interval_sum_fixed_d4.bin`
- `val_ids_interval_sum_fixed_d4.bin`
- `train_context_mask_interval_sum_fixed_d4.bin`
- `val_context_mask_interval_sum_fixed_d4.bin`
- `meta_interval_sum_fixed_d4.pkl`

Optional held-out eval length generation:

```bash
.venv/bin/python data/interval-sum/prepare.py \
  --digit_width=4 \
  --digit_width_mode=fixed \
  --min_sequence_tokens=2 \
  --max_sequence_tokens=32 \
  --eval_sequence_tokens=48 \
  --num_train=1000000 \
  --num_val=10000 \
  --overwrite
```

(`interval_sum` requires `min_sequence_tokens >= 2` because each example needs both `R` markers.)

With `--eval_sequence_tokens`, the base train/val splits are still emitted at the configured
`min_sequence_tokens..max_sequence_tokens` range, and additional eval files are written with
length-specific suffixes such as:

- `val_ids_interval_sum_fixed_d4_eval@l48.bin`
- `val_context_mask_interval_sum_fixed_d4_eval@l48.bin`
- `meta_interval_sum_fixed_d4_eval@l48.pkl`

Multiple lengths are supported via CSV (deduplicated and sorted), for example
`--eval_sequence_tokens=33,40,48`.
Each eval length must be greater than `--max_sequence_tokens` so the split is actually held out.

## Train

Example launch:

```bash
.venv/bin/python -m nanogptpro.train_adam_arithmetics \
  config/interval_sum/train_gpt_mha_rope_small_adam_interval_sum_l32_d4_single_gpu.py
```

The length-specific configs expect matching dataset directories such as
`interval-sum-l32` or `interval-sum-l64`. Prepare those by passing an explicit
output directory, for example `--output_dir=data/interval-sum-l32`.

Ordinary train/val runs use online generation by default. Online interval-sum
uses the same task text, tokenizer metadata, padding, and contextual loss mask
as this `prepare.py`; length-style dataset names such as `interval-sum-l32`
infer `min_sequence_tokens=max_sequence_tokens=32`. Pass
`--data_source=prepared` to read prepared `.bin` artifacts or held-out
`eval_sequence_tokens` splits.
