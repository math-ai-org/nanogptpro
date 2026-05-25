# openwebtext dataset

`python -m nanogptpro.train_adam_openwebtext` defaults to `corpus_format=text`, which loads raw
OpenWebText documents from the local Hugging Face datasets cache and tokenizes
them in the dataloader without network access. Set
`--text_local_files_only=False` to allow downloads, and `--text_streaming=True`
when remote streaming is desired. Run `prepare.py` only when you want `.bin`
files for `corpus_format=tokenized` training.

after running `prepare.py` (preprocess) we get:

- train.bin is ~17GB, val.bin ~8.5MB
- train has ~9B tokens
- val has ~4M tokens

this came from 8,013,769 documents in total.

Token boundaries:

- For `--tokenizer=gpt2`, each document is wrapped with BOS...EOS
  (`<|beginoftext|>...<|endoftext|>`). This repo adds a distinct GPT-2 BOS token id so boundaries between
  documents become adjacent EOS/BOS tokens (`<|endoftext|><|beginoftext|>`).
- During `.bin` writing, we optionally apply aligned streaming packing ("Fit or Cut") to avoid short
  orphan tail fragments at the start of a training block.
  - Disabled by default: `--no-fit_or_cut` (enable with `--fit_or_cut`).
  - Controlled by `--block_size` (model context window) and `--fit_or_cut_threshold` (minimum viable
    fragment length).
  - With `TrainAdamConfigBase.corpus_format=tokenized` and
    `tokenized_packing=auto`, disabling preprocessing-time packing selects
    runtime Fit-or-Cut packing in the `DataLoader`.

Tokenizer options:

- Default: `--tokenizer=gpt2` (tiktoken GPT-2 BPE, `uint16` tokens)
  - By default, special token strings in document text are treated as literal text (no special-token parsing).
    Use `--allow_special_tokens` to parse them into special token IDs.
- Alternative: `--tokenizer=gpt-oss` (Hugging Face tokenizer, default: `openai/gpt-oss-120b`, `uint32` tokens)
  - Uses the official GPT-OSS BPE vocabulary (~200k tokens) and wraps each document with
    `<|startoftext|>...<|return|>`.
  - By default, document text is encoded without special-token parsing so raw
    `<|startoftext|>`/`<|return|>`/`<|endoftext|>` strings are treated as literal text. Use
    `--allow_special_tokens` to parse special token strings inside document text.
  - Override the tokenizer repo with `--hf_tokenizer_id=...`.
- Alternative: `--tokenizer=byteoss` (UTF-8 bytes + GPT-OSS special tokens, `uint32` tokens)
  - By default, document text is encoded as raw UTF-8 bytes (no `<|...|>` special-token parsing), so raw
    `<|startoftext|>`/`<|return|>`/`<|endoftext|>` strings are treated as literal text. Use
    `--allow_special_tokens` to parse special token strings inside document text.
  - Training scripts default to `byteoss_vocab=compact`, remapping special token IDs so the model uses
    `vocab_size=277` (smaller embedding/LM head). Use `--byteoss_vocab=sparse` to keep `vocab_size=200019`.

references:

- OpenAI's WebText dataset is discussed in the
  [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset
