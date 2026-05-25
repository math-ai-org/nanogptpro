from __future__ import annotations

import os

from nanogptpro.train_adam_config import ArithmeticsTrainConfig

_batch_size: int = 64
_digit_width: int = 4
_global_batch_size: int = 512

_ddp: bool = int(os.environ.get("RANK", "-1")) != -1
_world_size: int = int(os.environ.get("WORLD_SIZE", "1")) if _ddp else 1
if _world_size != 1:
    raise ValueError(f"This config is for single-GPU runs (WORLD_SIZE=1), got WORLD_SIZE={_world_size}. ")

if _global_batch_size % _batch_size != 0:
    raise ValueError(f"global_batch_size={_global_batch_size} must be divisible by batch_size={_batch_size}.")
_gradient_accumulation_steps: int = _global_batch_size // _batch_size

CONFIG = ArithmeticsTrainConfig(
    # Wandb configs
    wandb_log=True,
    wandb_project="nanogpt-pro",
    # Tokenizer configs
    tokenizer="byteoss",
    byteoss_vocab="compact",
    # Data configs
    dataset="arithmetics-addition-fixed_digit",
    digit_width=_digit_width,
    batch_size=_batch_size,
    global_batch_size=_global_batch_size,
    gradient_accumulation_steps=_gradient_accumulation_steps,
    # Model configs (gpt-mha-rope-small)
    num_hidden_layers=12,
    num_attention_heads=6,
    hidden_size=768,
    head_dim=128,
    dropout=0.0,
    bias=False,
    using_groupnorm=False,
    use_qk_rmsnorm=True,
    # Embedding and hidden init
    embedding_init_std=0.02,
    hidden_init_std_factor=0.5,
    # Parameterization
    parameterization="widthmuP",
    hidden_size_base=1024,
    num_hidden_layers_base=24,
    embedding_lr_multiplier=1.0,
    # Training configs
    block_size=26,  # digit_width=4 default; overridden by dataset meta_d4.pkl if different
    max_iters=2000,
    lr_decay_iters=2000,
    eval_interval=100,
    eval_iters=100,
    log_interval=10,
    save_checkpoints=True,
    keep_last_checkpoints=3,
    # Optimizer configs
    optimizer_name="adamw",
    learning_rate_base=1e-4,
    min_lr_base=5e-6,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    decay_lr=True,
    warmup_iters=200,
    schedule="cosine",
    # System configs
    compile=False,
    model_type="DDL-gpt-mha-rope-CC-EC-lambda-accelerated",
)

globals().update(CONFIG.model_dump(exclude_unset=True))
