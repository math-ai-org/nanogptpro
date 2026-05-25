# Modern LLM Pre-training Data Specification for nanoGPT Pro

**Version**: 2.0 (NanoGPT Specialized)

**Scope**: Fixed Context Window Architectures (nanoGPT-style)

**Core Philosophy**: "Upstream Alignment, Downstream Simplicity" — Solve boundary issues during data packing to maintain a stateless, standard training loop.

---

## Preprocessing Output Defaults

Arithmetic dataset preprocessing entrypoints in `data/*/prepare.py` default to writing
generated artifacts under the same data root that arithmetic training configs read.

Default data root:

- `data/<dataset-name>`

Set `NANOGPT_DATA_ROOT` to use scratch for both preprocessing and training:

- `export NANOGPT_DATA_ROOT=/scratch/gpfs/ARORA/$USER/nanogptpro_mod/data`

Override a single preprocessing run with `--output_dir=/custom/path`.

Generated `.bin` / `.pkl` artifacts are ignored by git, including length-specific
directories such as `data/interval-sum-l32`.

---

## 1. Logical Data Unit (Semantic & Safety)

### 1.1 Structural Definition
Every document $D$ is encapsulated in a strictly closed structure before packing:

$$
T = [\text{BOT}_{\text{spec}}] + \text{Tokenize}(D) + [\text{EOT}_{\text{spec}}]
$$

### 1.2 Token Space Segregation (Critical Safety)
To prevent "Control Token Ambiguity" (where the model cannot output the text "[BOT]"), strict segregation is enforced during Tokenization:

* **Control Tokens (Special IDs)**:
* **IDs**: `BOT_ID` (e.g., 128000), `EOT_ID` (e.g., 128001).
* **Injection**: Inserted **only** by the Data Packer logic.
* **Function**: Semantic boundaries; used for Loss Masking.
* **Content Strings (Text IDs)**:
* **IDs**: Sequence of standard tokens for `[` `B` `O` `T` `]`.
* **Injection**: Parsed from raw text (e.g., GitHub code, documentation).
* **Tokenizer Setting**: `allowed_special={}` (Disallow auto-parsing of special tokens in body text; override with
  `--allow_special_tokens` in preprocessing scripts when the corpus intentionally includes control tokens).

---

## 2. Aligned Streaming & Packing (The "Fit or Cut" Protocol)

To eliminate "Orphan Fragments" (short, context-less tails at the start of a block) that destabilize
training, we employ **Fit-or-Cut** aligned packing.

This repo supports several places to assemble dense training batches:

- **Strategy 0 (Text corpus tokenization, text-training default)**: load raw text
  documents from the local Hugging Face datasets cache and tokenize them inside
  the training dataloader.
  - Pros: no `.bin` preprocessing step, no network access by default when the
    dataset cache is already populated, tokenizer changes take effect
    immediately, and this matches the nanochat-style default training path.
  - Cons: higher CPU/cache I/O load during training, and validation uses a
    deterministic document-stride holdout rather than prewritten validation files.
  - Contract: `TrainAdamConfigBase.corpus_format="text"` selects this path for
    OpenWebText and FineWeb-EDU. The loader manually wraps each document with
    tokenizer BOS/EOS IDs, treats special-token-looking text as ordinary corpus
    text, uses the same document scheduler as runtime tokenized mode, and uses
    the same Fit-or-Cut packing rule. Document suffixes are preserved across row
    and batch boundaries unless `fit_or_cut_threshold` drops a short wrap
    fragment; when that truncation path would keep only a short document prefix
    in the current block, the loader pads the prefix slot with the resolved
    packing padding token id and starts the document in the next block. Padding
    targets are ignored.
    Defaults are `text_local_files_only=True` and `text_streaming=False`; set
    `text_local_files_only=False` to permit downloads and `text_streaming=True`
    to opt into remote streaming.
    `dataloader_cpu_workers` defaults to 4 per training process/rank and
    tokenizes prefetched documents with an ordered CPU worker pool.
- **Strategy 0b (Online arithmetics generation)**: the default arithmetic
  training path. Create fixed/variable addition, `interval-sum`, `last-sum`,
  or `interval-last` examples inside `nanogptpro.train_adam_arithmetics` with
  `data_source=online`.
  - Pros: no `.bin` preprocessing step, no dataset storage, tokenizer and digit
    width / sequence-length changes take effect immediately, and short debug
    runs can start from a config file alone.
  - Cons: prepared `.bin` rows remain the fastest path for fixed finite
    train/validation sets, `data_loader=random_example`, and held-out
    `eval_sequence_tokens` eval splits.
  - Contract: the online generator emits complete `[BOS] context text [EOS]`
    examples, pads to the same shape calculation used by each task's
    `prepare.py`, and applies the same context mask so prompt/context targets
    are ignored. Stateless mode derives train and eval batches from
    `data_seed`/`eval_seed`, split, rank, and batch index; stateful mode
    checkpoints the per-rank RNG state in `rng_state*.pth`.
- **Strategy A (Static pre-processing)**: apply Fit-or-Cut during `.bin` generation.
  - Pros: fastest training-time data loading; simplest `DataLoader`.
  - Cons: the resulting `.bin` becomes **block_size-coupled**; changing training `block_size` requires
    re-packing the dataset.
  - Contract: `meta.pkl["packing"]["enabled"] == True` and
    `meta.pkl["packing"]["block_size"] == train_config.block_size` (enforced at runtime).
  - Sidecars: optional for this path; the static token-stream loader does not require
    `.doc_bounds.npy` files. When padding is inserted before a document, document
    bounds may contain gaps that correspond to ignored padding tokens.
- **Strategy B (Dynamic re-packing for `.bin` streams)**: store a raw BOS...EOS document stream on disk and apply
  Fit-or-Cut in the `DataLoader` at runtime.
  - Pros: training `block_size` can change without re-tokenizing/re-packing on disk.
  - Cons: higher CPU load during data loading; may become a bottleneck at scale.
  - CPU workers: `dataloader_cpu_workers` defaults to 4 per training
    process/rank and enables ordered prefetch workers for runtime document
    materialization while preserving the same checkpoint state contract.
  - Implementation note: preprocessing writes document-bound sidecars that
    index raw document `[start, end)` spans inside each token stream shard.
    Those sidecars are `block_size`-agnostic and are required by the runtime
    loader, because recovering boundaries from raw BOS/EOS tokens alone is
    ambiguous once document text can contain those special tokens.
    Runtime tokenized DDP assigns complete documents by shared scheduled position
    (`scheduled_position % world_size == rank`) rather than by cutting shards
    into per-rank token segments, so rank boundaries do not drop documents.
    The same `document_schedule_chunk_size` must be used by text and runtime
    tokenized loaders when comparing their shuffled document order.
    Text training caps `fit_or_cut_threshold` to `block_size - 1` for runtime
    Fit-or-Cut paths, so small context-window debug runs can use the default
    threshold.
    The runtime loader also validates sidecars against the raw token stream, so
    stale or corrupted sidecars fail fast instead of silently changing the
    packed batches. Dataset writers also use a small publish-state / rollback
    protocol so interrupted `.bin` + `.doc_bounds.npy` updates recover to the
    last consistent pair before the next read or overwrite, and preprocessing
    invalidates stale `meta.pkl` before rewriting outputs so interrupted
    overwrites fail fast instead of mixing old metadata with new data files.
    Training also refuses to load datasets when `meta.pkl` is missing, even if
    only legacy `.bin` files are present, so interrupted or stale dataset
    states cannot silently fall back to loader defaults. Training also requires
    `meta.pkl` to satisfy the current strict metadata schema for every loader
    mode, including tokenizer, token dtype, packing, document-boundary, and
    FineWeb shard/datafile fields. Missing fields are treated as an unsupported
    legacy dataset state instead of being filled from loader defaults.
    Preprocessing leaves an explicit metadata invalidation marker until the new
    `meta.pkl` has been published, so crashed reruns fail fast after deleting
    metadata but before rewriting it.
    FineWeb-EDU additionally records shard-count metadata and prunes stale
    shard artifacts after successful rewrites so shorter reruns do not leave
    old `fineweb_train_*.bin` shards behind for training to pick up. The
    FineWeb training entrypoint consumes that shard metadata for both train and
    validation splits, so datasets with multiple `fineweb_val_*.bin` shards
    now evaluate correctly instead of assuming a single validation shard.
  - Contract: `meta.pkl["packing"]["enabled"] == False` and
    `TrainAdamConfigBase.tokenized_packing="auto"` (the default) selects the
    runtime packer.

### 2.1 The "Fit or Cut" Algorithm

Let `Block_Size` be the model's context window (e.g., 4096).
Let `Cursor` be the current fill level of the active block (0 to 4095).
Let `Threshold` be the minimum viable fragment length (e.g., 100 tokens).
Let `Prefix_Slot = Block_Size - Cursor`.

For each incoming Document $D$ of length $L$:

1.  **Calculate Remainder**: Determine where $D$ would end if written continuously.
 
$$
R = (Cursor + L) \pmod{Block\_Size}
$$
    *(Note: If $D$ spans multiple blocks, $R$ is the fill level of the final block)*

2.  **Decision Logic**:

* **Scenario A: Clean Fit / No Wrap / Viable Wrap Tail** ($R == 0$ or
  $R \ge Threshold$):
* **Action**: Write $D$ fully.
* **Result**: No short wrap tail must be dropped; if $D$ wraps, the final
  fragment is long enough to provide valid gradients.
* **Scenario B: Short Wrap Tail With Viable Current-Block Prefix** (writing $D$
  wraps, $0 < R < Threshold$, and (`Cursor == 0` or
  `Prefix_Slot >= Threshold`)):
* **Action**: Drop the last $R$ tokens of $D$.
* **Result**: The current block is filled exactly to the end.
* **Benefit**: The **Next Block** starts at Index 0 with a fresh `[BOT]`.
* **Scenario C: Short Prefix Slot on the Short-Tail Path** (writing $D$ wraps,
  $0 < R < Threshold$, `Cursor > 0`, and `Prefix_Slot < Threshold`):
* **Action**: Fill the remaining `Prefix_Slot` tokens with the resolved packing
  padding token id, start the document at the beginning of the next block (where
  `Cursor == 0`), then re-apply the Fit-or-Cut decision.
* **Result**: The document is preserved when truncation would leave only a short,
  context-poor prefix before the block boundary.
* **Loss**: Padding targets are ignored; positions where either the input or target is
  the padding token are rewritten to `ignore_index`.

### 2.2 Outcome: The "Golden Start"
By truncating short wrap tails and padding short prefix slots on that same
short-tail path, we maximize the occurrence of blocks where:
`Block[0] == [BOT]`

---

## 3. Training Architecture (nanoGPT Standard)

### 3.1 Attention Mechanism
* **Operator**: `torch.nn.functional.scaled_dot_product_attention` (SDPA).
* **Mode**: Standard Causal (Lower Triangular Mask).
* **Padding**: Packing may insert padding tokens to seal short prefix slots, but
  targets touching padding are ignored.
* **Constraint**: Dense SDPA remains sufficient; no custom block-masking or Varlen
  logic is required for these padded packing slots.

---

## 4. Loss Calculation (Loss Masking)

Despite the clean data, we may optionally enforce "Semantic Causality" by masking certain
cross-document transitions (e.g. `[EOT] -> [BOT]`).
By default, we **do compute** loss on `[EOT]` predicting `[BOT]` when `[BOT]` is the true target, but
this is configurable (see §4.2).

### 4.1 Masking Logic Table

| Token Position | Input ID | Target ID | Loss Status | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Doc Start** | `[EOT]` (Prev) | `[BOT]` (Curr) | **COMPUTE (default)** / **IGNORE (-1)** | `[EOT] -> [BOT]` is a deterministic boundary transition; masking is optional via `ignore_doc_start_loss=True`. |
| **Body Start** | `[BOT]` | `Word_1` | **COMPUTE** | Critical: Learning to initiate content from a start signal. |
| **Body Content** | `Word_t` | `Word_{t+1}` | **COMPUTE** | Standard language modeling. |
| **Doc End** | `Word_n` | `[EOT]` | **COMPUTE** | Critical: Learning when to stop. |

### 4.2 Implementation
```python
# Training-time masking rewrites the target stream with the model's
# ignore_index (-1).
# In this repo, the doc-start token is the tokenizer's BOS id.

if ignore_doc_start_loss and bos_token_id is not None:
    targets[targets == bos_token_id] = -1

if padding_token_id is not None:
    targets[(inputs == padding_token_id) | (targets == padding_token_id)] = -1
```

---

## 5. Executive Summary

1. **Pre-processing**: Use **Aligned Packing**. Truncate short wrap tails, but pad
   short prefix slots when padding preserves the incoming document more cleanly.
2. **Training**: Use standard **SDPA** with causal masking. No custom block-masking required.
3. **Safety**: Distinct IDs for Control `[BOT]` vs String `"[BOT]"`.
4. **Loss**: By default, compute loss on `[EOT] -> [BOT]`; optionally mask `[BOT]`
   targets via `ignore_doc_start_loss=True`, and always ignore packing padding
   targets.
