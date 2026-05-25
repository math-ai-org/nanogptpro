# NanoGPT Pro

NanoGPT Pro ships as a compiled training and inference runtime. Install the
binary release wheel, then use this repository for the assets that go with it:

- `nanogptpro/config/`: the JSON training configuration catalog.
- `data_recipes/`: dataset preprocessing entrypoints and the data-format
  specification.
- `lm-evaluation-harness/`: bundled checkpoint evaluation harness with the
  NanoGPT Pro bridge.

## Requirements

- Linux x86_64 for the release wheel
- Python 3.14
- `uv`
- A CUDA-capable PyTorch environment for GPU training or GPU inference
- A NanoGPT Pro checkpoint saved in Hugging Face format for inference or
  evaluation

Generated data, logs, and checkpoints are intentionally not committed. Put
large datasets and outputs on local scratch storage when available.

## Architectures

NanoGPT Pro is organized around **Fast Weight Attention (FWA)**, a unifying
framework that casts recurrent state writes as online learning rules. DeltaNet,
Mamba, Lightning Attention, and the project's Falcon family are concrete
implementations of this framework, trained alongside standard softmax-attention
baselines through one config-driven pipeline:

- **Softmax attention:** MHA + RoPE, MHA + ALiBi, MHA + GRAPE, GQA, MQA, MLA,
  TPA, and related attention variants.
- **FWA · DeltaNet:** DeltaNet and Gated DeltaNet, with chunked Triton and FLA
  kernels and context-dependent `beta`/`lambda`/`eta` gating.
- **FWA · Linear / SSM:** Mamba 2, Mamba 3, and Lightning Attention.
- **FWA · Falcon:** Falcon-1/2/3 and Falcon-1A/2A/3A variants.

The matching training recipes live under `nanogptpro/config/`.

## Install

Download the release wheel and install it into a Python 3.14 environment:

```bash
git clone https://github.com/math-ai-org/nanogptpro.git
cd nanogptpro

curl -LO https://github.com/math-ai-org/nanogptpro/releases/download/v1.0.0/nanogptpro-1.0.0-cp314-cp314-linux_x86_64.whl

uv venv --python 3.14 .venv
source .venv/bin/activate
uv pip install ./nanogptpro-1.0.0-cp314-cp314-linux_x86_64.whl
```

Check that the runtime is available:

```bash
python -c "import nanogptpro; print(nanogptpro.__version__)"
python -c "from nanogptpro.model_registry import list_model_ids; print('\n'.join(list_model_ids()[:20]))"
nanogptpro-generate --help
nanogptpro-train-openwebtext --help
```

Run the commands from a work directory outside the repository root if you add
local Python files, so nothing shadows the installed package on `sys.path`.

## Inference

Run generation from a local checkpoint directory:

```bash
nanogptpro-generate \
  --model-id gpt-mha-rope \
  --checkpoint /path/to/checkpoint \
  --prompt "The theorem states that" \
  --max-new-tokens 128 \
  --device cuda \
  --dtype bfloat16
```

For a CPU smoke test, use a short generation and float32:

```bash
nanogptpro-generate \
  --model-id gpt-mha-rope \
  --checkpoint /path/to/checkpoint \
  --prompt "Hello" \
  --max-new-tokens 16 \
  --device cpu \
  --dtype float32
```

The default `--tokenizer auto` reads tokenizer hints from checkpoint
`config.json` when available. If the checkpoint config does not identify the
tokenizer, pass it explicitly:

```bash
# GPT-2/tiktoken style tokenizer
nanogptpro-generate \
  --model-id gpt-mha-rope \
  --checkpoint /path/to/checkpoint \
  --prompt "Hello" \
  --tokenizer gpt2 \
  --encoding gpt2

# Hugging Face tokenizer directory or model id
nanogptpro-generate \
  --model-id gpt-mha-rope \
  --checkpoint /path/to/checkpoint \
  --prompt "Hello" \
  --tokenizer hf \
  --hf-tokenizer-id /path/to/tokenizer \
  --local-files-only
```

You can also call the Python API directly:

```python
import torch

from nanogptpro.llm import LLM
from nanogptpro.sampling_params import SamplingParams
from nanogptpro.tokenizer import TokenizerConfig

llm = LLM.from_checkpoint(
    model_id="gpt-mha-rope",
    checkpoint="/path/to/checkpoint",
    tokenizer=TokenizerConfig(tokenizer="auto"),
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
)

print(llm.generate_one("The theorem states that", SamplingParams(max_new_tokens=128)))
```

## Training

The wheel exposes these training commands:

- `nanogptpro-train-openwebtext`
- `nanogptpro-train-text-dataset`
- `nanogptpro-train-arithmetics`
- `nanogptpro-train-var-arithmetics`

Each command takes one positional `CONFIG` argument: a catalog name compiled
into the wheel, or an explicit path to a training-config JSON record from
`nanogptpro/config/`. See `nanogptpro/config/README.md` for the catalog
contract. `--name=value` arguments override individual config values for one
run:

```bash
nanogptpro-train-openwebtext \
  train_gpt_mha_rope_small_adam_50BT_ctx1024_80g4 \
  --wandb_log=False
```

The same run from the checked-in JSON record:

```bash
nanogptpro-train-openwebtext \
  nanogptpro/config/text/train_gpt_mha_rope_small_adam_50BT_ctx1024_80g4.json \
  --wandb_log=False
```

Generated datasets default to the untracked `data_files/` root. Set
`NANOGPT_DATA_ROOT` to keep both preprocessing outputs and training inputs on
scratch storage:

```bash
export NANOGPT_DATA_ROOT=/scratch/$USER/nanogptpro/data_files
```

Training writes run artifacts under the `output` directory by default.

## Prepare Data

NanoGPT Pro text training can read raw Hugging Face datasets, or you can
pre-tokenize datasets into local artifacts with the scripts under
`data_recipes/`. The preprocessing entrypoints import helpers from the
installed wheel, so run them inside the environment created above.

OpenWebText preprocessing example:

```bash
python data_recipes/openwebtext/prepare.py
```

FineWeb-EDU 10B preprocessing example:

```bash
python data_recipes/fineweb-edu/fineweb-edu.py --version=10B
```

OpenWebText and FineWeb-EDU are large datasets. Use `--max_docs=...` on the
preparation scripts for a small local preprocessing smoke test, and see each
recipe's `README.md` for the full argument reference.

Then train with the matching command:

```bash
nanogptpro-train-text-dataset \
  train_gpt_mha_rope_small_adam_50BT_ctx1024_80g4 \
  --dataset=fineweb-edu10B \
  --corpus_format=tokenized \
  --wandb_log=False
```

## Evaluate a Checkpoint

Install the bundled evaluation harness in the same environment you use for
inference:

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

## Checkpoint Notes

- Prefer checkpoints whose `config.json` contains `architectures` and
  `nanogptpro_model_type`.
- Pass an explicit `--model-id` when running inference on older checkpoints
  without `nanogptpro_model_type`.
- Some LM Evaluation Harness tasks download datasets on first use.

## Citation

If you use NanoGPT Pro in research, please cite:

```bibtex
@misc{nanogptpro2026,
  title   = {NanoGPT Pro: A Multi-Architecture NanoGPT Training and Evaluation Suite},
  author  = {Team Math-AI},
  journal = {math-ai-org.github.io},
  year    = {2026},
  url     = {https://github.com/math-ai-org/nanogptpro}
}
```
