# Active Training Configurations

This directory contains the supported training configuration catalog.

Each JSON file stores separate `checkpointing`, `data`, `model`,
`optimization`, `tokenization`, and `runtime` objects. The package validates
every file during import. The loader never executes configuration code.
Every current record uses `nanogptpro.training_config.v4`; obsolete formats are
rejected rather than reinterpreted.

Use the public API:

```python
from nanogptpro.config import get_training_config
from nanogptpro.train import train

config = get_training_config("train_gpt_mha_rope_small_adam_50BT_ctx1024_80g4")
result = train(
    config.config,
    config.runtime,
    train_corpus=train_corpus,
    val_corpus=val_corpus,
)
```

Standalone training commands accept compatible direct and DDP names. They also
accept an explicit path to a complete compatible JSON record:

```bash
nanogptpro-train-text-dataset \
    train_gpt_mha_rope_small_adam_50BT_ctx1024_80g4
```

Python configuration files are intentionally unsupported and are never
executed. Use `--name=value` arguments for one-run overrides.

List runnable public names deterministically:

```bash
python -c 'from nanogptpro.config import list_training_configs as names; '\
'print(*names(available_only=True), sep="\n")'
```

`get_training_config()` rejects entries whose model implementation is absent.
`TRAINING_CONFIGS` keeps those entries visible for inspection.

The loader sorts each relative JSON path before it loads the files.

The record name is the public configuration identity. Storage directories can
group related records without changing their public names.

Each storage filename and public name includes `ctx{block_size}`.
The marker must match `data.block_size`.

The `tokenization` object stores `encoding`, vocabulary size, and special-token
IDs. The loader derives `tokenizer_kind` and `hf_tokenizer_id` from `encoding`.

The `data` object stores data shape, seed, and document packing controls.

The `model` object stores `torch_compile` as the switch for `torch.compile()`
and the explicit `flash_attention_impl` identity selection. Catalog entries use
`builtin`; custom SM90 variable-length runs may select `fa3` after installing
the pinned source extra.

The `optimization` object stores micro-batch, evaluation, loss masking,
parameterization, optimizer, scheduler, and sequence packing controls.
`micro_batch_size` controls each rank's device batch and `global_batch_size`
controls sequences per optimizer update across all ranks. The trainer derives
gradient accumulation from those values and the runtime world size; it is not
a configuration field.

Standalone training uses `output` as the default runtime output directory.
The `checkpointing` object provides `checkpoint_interval` and
`keep_last_checkpoints` to standalone commands. The commands always commit the
final checkpoint. Resume accepts an exact run directory with current v19
artifacts.

The active package is self-contained. The loader and active catalog tests do
not read files under `archive/`.

The catalog contains 74 text configurations and nine arithmetic configurations.

The 3.3B `gpt-mha-rope` configuration uses dense FSDP2 plan version 3. It
requires eight ranks and uses BF16 parameter all-gathers. The catalog does not
provide the similar GQA configuration. NanoGPT Pro has no registered sharded
GQA plan, and replicated DDP exceeds the safe per-rank memory contract.

The `moe` directory contains matched medium 50BT DeepSeekMoE and LatentMoE
DeepEP 2.1 configurations. They match the dense `ctx1024` baseline outside the
required MoE fields. Each requires one eight-rank SM90 NVLink node and uses the
registered FSDP2 expert plan version 3 in `direct` mode with
`deepep_transport="nvlink"`. Multi-node jobs may also use NVLink when every EP
group remains node-local; any EP group that spans nodes, including hybrid mode,
must use the `gin` transport. The LatentMoE recipe uses a four-times routed-width
compression ratio and increases routed experts and active routes to preserve
matched active compute.

The `arithmetics` directory stores fixed-digit and variable-digit addition
training configurations for the small `gpt-mha-rope` model. These
configurations require ByteOSS tokenization and one-example-per-row sequence
packing.

Each configuration selects one explicit device. Restored macOS configurations
use CPU and the `_cpu` name suffix. They do not select MPS at run time.

The `fast-weight-attention` directory groups 54 fast-weight attention
configurations as 27 matched small and medium recipes.
