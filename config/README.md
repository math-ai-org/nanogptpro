# README

These configs target a fixed `global_batch_size` (sequences per optimizer step across all ranks)
so each W&B step corresponds to the same number of tokens regardless of whether you're running
DDP or a single process.

Formulas:

- Tokens / step: `global_batch_size * block_size`
- Total tokens: `global_batch_size * block_size * max_iters`

Examples for ~50B tokens at `global_batch_size=480`:

1. Context length 4096: `480 * 4096 * 25000`
2. Context length 1024: `480 * 1024 * 100000`
