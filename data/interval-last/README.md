# Interval-last dataset

This directory contains the dedicated mixed-task family where each example is either:

- `interval_sum`
- `last_sum`

This is an independent dataset entrypoint. There is no generic `mixed` task-mode surface anymore.

Digit-width variability is controlled separately by:

- `digit_width_mode=fixed`
- `digit_width_mode=variable`

## Generate data

```bash
.venv/bin/python data/interval-last/prepare.py \
  --digit_width=4 \
  --digit_width_mode=fixed \
  --min_sequence_tokens=5 \
  --max_sequence_tokens=8 \
  --num_train=1000000 \
  --num_val=10000 \
  --overwrite
```

By default, this writes generated files to:

- `data/interval_last/`

Set `NANOGPT_DATA_ROOT` to write and train from scratch instead:

- `export NANOGPT_DATA_ROOT=/scratch/gpfs/ARORA/$USER/nanogptpro_mod/data`

You can override a single run with:

- `--output_dir=/custom/path`

For the canonical fixed-width setup above, the written files use suffix:

- `interval_last_fixed_d4`

## Train

```bash
.venv/bin/python -m nanogptpro.train_adam_arithmetics \
  config/interval_last/train_gpt_mha_rope_small_adam_interval_last_l32_d4_single_gpu.py
```

The length-specific configs expect matching dataset directories such as
`interval-last-l32` or `interval-last-l64`. Prepare those by passing an explicit
output directory, for example `--output_dir=data/interval-last-l32`.

Ordinary train/val runs use online generation by default. Online interval-last
uses the same mixed `interval_sum` / `last_sum` task sampling, tokenizer
metadata, padding, and contextual loss mask as this `prepare.py`; length-style
dataset names such as `interval-last-l64` infer
`min_sequence_tokens=max_sequence_tokens=64`. Pass `--data_source=prepared` to
read prepared `.bin` artifacts or held-out `eval_sequence_tokens` splits.
