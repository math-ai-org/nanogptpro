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
.venv/bin/python data_recipes/interval-last/prepare.py \
  --digit_width=4 \
  --digit_width_mode=fixed \
  --min_sequence_tokens=5 \
  --max_sequence_tokens=8 \
  --num_train=1000000 \
  --num_val=10000 \
  --overwrite
```

By default, this writes generated files to:

- `data_files/interval_last/`

Set `NANOGPT_DATA_ROOT` to write and train from scratch instead:

- `export NANOGPT_DATA_ROOT=/scratch/gpfs/ARORA/$USER/nanogptpro_mod/data_files`

You can override a single run with:

- `--output_dir=/custom/path`

For the canonical fixed-width setup above, the written files use suffix:

- `interval_last_fixed_d4`

## Train

```bash
TRAINING_CONFIG="arithmetics/fixed_digit/"\
"train_gpt_mha_rope_small_adam_arithmetics_addition_fixed_digit_4_ctx26_single_gpu"
.venv/bin/python -m nanogptpro.train_adam_arithmetics "${TRAINING_CONFIG}" \
  --dataset=interval-last-l32 \
  --digit_width=4 \
  --dataset_suffix=interval_last_fixed_d4
```

For prepared data, the length-specific `--dataset` override resolves a matching
dataset directory such as `interval-last-l32` or `interval-last-l64`. Prepare
that directory explicitly, for example
`--output_dir=data_files/interval-last-l32`.

Ordinary train/val runs use online generation by default. Online interval-last
uses the same mixed `interval_sum` / `last_sum` task sampling, tokenizer
metadata, padding, and contextual loss mask as this `prepare.py`; length-style
dataset names such as `interval-last-l64` infer
`min_sequence_tokens=max_sequence_tokens=64`. Pass `--data_source=prepared` to
read prepared `.bin` artifacts or held-out `eval_sequence_tokens` splits.
