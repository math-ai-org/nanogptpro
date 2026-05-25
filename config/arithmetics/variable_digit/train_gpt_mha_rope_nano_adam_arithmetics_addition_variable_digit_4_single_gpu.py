from __future__ import annotations

from nanogptpro.train_adam_config import ArithmeticsTrainConfig, single_gpu_gradient_accumulation_steps

_batch_size: int = 64
_digit_width: int = 4
_global_batch_size: int = 512

_gradient_accumulation_steps: int = single_gpu_gradient_accumulation_steps(
    batch_size=_batch_size,
    global_batch_size=_global_batch_size,
)

CONFIG = ArithmeticsTrainConfig(
    # Wandb configs
    wandb_log=True,
    wandb_project="nanogpt-pro",
    # Tokenizer configs
    tokenizer="byteoss",
    byteoss_vocab="compact",
    # Data configs
    dataset="arithmetics-addition-variable_digit",
    digit_width=_digit_width,
    batch_size=_batch_size,
    global_batch_size=_global_batch_size,
    gradient_accumulation_steps=_gradient_accumulation_steps,
    # Model configs (gpt-mha-rope-nano)
    num_hidden_layers=4,
    num_attention_heads=2,
    hidden_size=256,
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
    compile=True,
    model_type="gpt-mha-rope",
)

globals().update(CONFIG.model_dump(exclude_unset=True))
