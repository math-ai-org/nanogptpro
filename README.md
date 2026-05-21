# NanoGPT Pro

NanoGPT Pro ships a compiled Python package for training, running, and
evaluating NanoGPT checkpoints.

Use this repository to install the published wheel, run bundled training
configs, prepare datasets, and evaluate checkpoints. You do not need a source
checkout for normal release use.

## Requirements

- Linux x86_64
- Python 3.14
- `uv`
- A CUDA-capable PyTorch environment for training
- A NanoGPT Pro checkpoint saved in Hugging Face format for evaluation

For GPU training or evaluation, install a PyTorch build that matches your CUDA
environment before installing the wheel.

## Install

Download the `nanogptpro` wheel from the matching GitHub Release, then install
it in a fresh environment:

```bash
uv venv --python 3.14 .venv
source .venv/bin/activate
uv pip install ./nanogptpro-1.0.0-cp314-cp314-linux_x86_64.whl
```

Check that the package is available:

```bash
python -c "import nanogptpro; print(nanogptpro.__version__)"
nanogptpro-generate --help
```

## Repository Layout

- `config/`: training configs. The examples below use `gpt-mha-rope`.
- `data/`: dataset preparation scripts and data-format documentation.
- `lm-evaluation-harness/`: bundled checkpoint evaluation harness.

Generated data, logs, and checkpoints are intentionally not committed. Put large
datasets and outputs on local scratch storage when available.

## Download or Prepare Data

NanoGPT Pro text training defaults to raw Hugging Face datasets. The first
command below permits a dataset download into the Hugging Face cache and trains
from raw text. Later runs can omit `--text_local_files_only=False` to stay
offline and reuse the cache.

```bash
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"

nanogptpro-train-openwebtext \
  config/train_gpt_mha_rope_small_adam_50BT_ctx1024_80g4.py \
  --data_path=data \
  --out_dir=out/gpt-mha-rope-openwebtext \
  --text_local_files_only=False \
  --wandb_log=False
```

For an explicit preprocessing step, write tokenized OpenWebText artifacts under
`data/openwebtext/`:

```bash
python data/openwebtext/prepare.py \
  --output_dir=data/openwebtext \
  --tokenizer=gpt2 \
  --fit_or_cut \
  --block_size=1024 \
  --overwrite
```

Then train from the pre-tokenized files:

```bash
nanogptpro-train-openwebtext \
  config/train_gpt_mha_rope_small_adam_50BT_ctx1024_80g4.py \
  --data_path=data \
  --corpus_format=tokenized \
  --out_dir=out/gpt-mha-rope-openwebtext-tokenized \
  --wandb_log=False
```

FineWeb-EDU uses the same model configs with the FineWeb training entrypoint.
For example, prepare the 10B sample:

```bash
python data/fineweb-edu/fineweb-edu.py \
  --version=10B \
  --output_dir=data/fineweb-edu/fineweb-edu10B \
  --tokenizer=gpt2 \
  --fit_or_cut \
  --block_size=1024
```

Then train from that tokenized shard set:

```bash
nanogptpro-train-finewebedu \
  config/train_gpt_mha_rope_small_adam_50BT_ctx1024_80g4.py \
  --data_path=data/fineweb-edu \
  --dataset=fineweb-edu10B \
  --corpus_format=tokenized \
  --out_dir=out/gpt-mha-rope-fineweb-edu10B \
  --wandb_log=False
```

OpenWebText and FineWeb-EDU are large datasets. Use `--max_docs=...` on the
preprocessing scripts for a small local smoke test.

## Train `gpt-mha-rope`

Start with the 50B-token, context-1024 config:

```bash
nanogptpro-train-openwebtext \
  config/train_gpt_mha_rope_small_adam_50BT_ctx1024_80g4.py \
  --data_path=data \
  --out_dir=out/gpt-mha-rope-openwebtext \
  --wandb_log=False
```

For a short smoke test on one GPU, override the schedule:

```bash
nanogptpro-train-openwebtext \
  config/train_gpt_mha_rope_small_adam_50BT_ctx1024_80g4.py \
  --data_path=data \
  --out_dir=out/smoke-gpt-mha-rope \
  --text_local_files_only=False \
  --max_iters=20 \
  --eval_interval=10 \
  --eval_iters=5 \
  --batch_size=1 \
  --global_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --compile=False \
  --wandb_log=False
```

For multi-GPU training, launch the module with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m nanogptpro.train_adam_openwebtext \
  config/train_gpt_mha_rope_small_adam_50BT_ctx1024_80g4.py \
  --data_path=data \
  --out_dir=out/gpt-mha-rope-openwebtext \
  --wandb_log=False
```

Swap in another `config/train_gpt_mha_rope_*.py` file for larger runs. The
configs are normal Python files, and any value can be overridden from the
command line with `--name=value`.

## Evaluate a Checkpoint

Install the bundled evaluation harness in the same environment:

```bash
source .venv/bin/activate
uv pip install -e "./lm-evaluation-harness[hf]"
```

Run the convenience wrapper with a local checkpoint directory:

```bash
bash lm-evaluation-harness/test.sh \
  /path/to/checkpoint \
  auto \
  0 \
  results/checkpoint \
  arc_easy,hellaswag
```

The checkpoint directory should contain a Hugging Face style `config.json` and
model weights. If `config.json` includes `nanogptpro_model_type`, leave the
second argument as `auto`. If that field is missing, pass the model id as the
second argument:

```bash
bash lm-evaluation-harness/test.sh \
  /path/to/checkpoint \
  gpt-mha-linear_attention_wd_ctxlambda \
  0 \
  results/checkpoint \
  arc_easy,hellaswag
```

If tokenizer information is not stored in the checkpoint config, pass the
tokenizer path or Hugging Face tokenizer id as the sixth argument:

```bash
bash lm-evaluation-harness/test.sh \
  /path/to/checkpoint \
  auto \
  0 \
  results/checkpoint \
  arc_easy,hellaswag \
  gpt2
```

You can also call `lm_eval` directly:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=/path/to/checkpoint \
  --tasks arc_easy,hellaswag \
  --batch_size auto \
  --output_path results/checkpoint \
  --num_fewshot 0
```

## Useful Commands

```bash
lm_eval --help
lm_eval --tasks list
nanogptpro-generate --help
nanogptpro-train-openwebtext --help
nanogptpro-train-finewebedu --help
```

## Checkpoint Notes

- Prefer checkpoints whose `config.json` contains `nanogptpro_model_type`.
- Pass an explicit model id when evaluating older checkpoints without that
  config field.
- Keep the `nanogptpro` wheel version and the evaluation harness checkout from
  the same release.
- Some LM Evaluation Harness tasks download datasets on first use.
