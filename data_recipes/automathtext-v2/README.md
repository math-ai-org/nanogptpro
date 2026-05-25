# AutoMathText-V2 dataset

`automathtext-v2.py` preprocesses
[OpenSQZ/AutoMathText-V2](https://huggingface.co/datasets/OpenSQZ/AutoMathText-V2)
into local tokenized `.bin` shards for `corpus_format=tokenized` training.

The training-time raw text path can also load AutoMathText-V2 directly from the
Hugging Face datasets cache with `corpus_format=text`; use this script only when
you want pinned offline `.bin` artifacts or maximum-throughput tokenized runs.

Example:

```bash
python data_recipes/automathtext-v2/automathtext-v2.py --config ultra
```

After preprocessing, the default output is:

- `data_files/automathtext-v2/automathtext-v2-ultra/meta.pkl`
- `data_files/automathtext-v2/automathtext-v2-ultra/automathtext_val_000000.bin`
- `data_files/automathtext-v2/automathtext-v2-ultra/automathtext_train_000001.bin`, ...

Supported shorthand configs:

- `--config=ultra` -> `automathtext-v2-ultra`
- `--config=high` -> `automathtext-v2-high`
- `--config=medium-high` -> `automathtext-v2-medium-high`
- `--config=low-medium-high` -> `automathtext-v2-low-medium-high`

Domain configs can be selected directly, for example:

```bash
python data_recipes/automathtext-v2/automathtext-v2.py --config math_web
python data_recipes/automathtext-v2/automathtext-v2.py --config reasoning_qa
```

Tokenizer options match FineWeb-EDU preprocessing:

- Default: `--tokenizer=gpt2`
- Alternative: `--tokenizer=gpt-oss`
- Alternative: `--tokenizer=byteoss`

To train from the tokenized artifacts:

```bash
torchrun --standalone --nproc_per_node=8 \
    -m nanogptpro.train_adam_text_dataset \
    train_gpt_mha_rope_medium_adam_50BT_ctx4096_80g8 \
    --data_path=data_files/automathtext-v2 \
    --dataset=automathtext-v2-ultra \
    --corpus_format=tokenized
```

Avoid `--config=refinecode` unless your pipeline separately fetches the external
code content referenced by that domain's metadata.
