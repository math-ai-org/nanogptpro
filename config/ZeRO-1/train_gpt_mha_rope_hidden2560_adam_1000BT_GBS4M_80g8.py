from nanogptpro.train_adam_config import TrainAdamConfigBase

_batch_size = 4

CONFIG = TrainAdamConfigBase(
    # Wandb configs
    wandb_log=True,
    wandb_project="nanogpt-pro",
    # Tokenizer configs
    tokenizer="gpt-oss",
    # Model configs
    num_hidden_layers=36,
    num_attention_heads=10,
    hidden_size=2560,
    head_dim=256,
    dropout=0.0,
    bias=False,
    using_groupnorm=False,
    use_qk_rmsnorm=True,
    rope_ratio=0.5,
    # Embedding and hidden init
    embedding_init_std=0.02,
    hidden_init_std_factor=0.5,
    # Parameterization
    parameterization="widthmuP",
    hidden_size_base=1024,
    num_hidden_layers_base=24,
    embedding_lr_multiplier=1.0,
    # Training configs
    seed=42,
    batch_size=_batch_size,
    global_batch_size=1024,
    block_size=4096,
    gradient_accumulation_steps=128 // _batch_size,
    max_iters=250000,
    lr_decay_iters=250000,
    eval_interval=1000,
    eval_iters=200,
    log_interval=10,
    save_checkpoints=True,
    keep_last_checkpoints=3,
    # Optimizer configs
    optimizer_name="adamw",
    learning_rate_base=3e-3,
    min_lr_base=1.5e-3,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    # ZeRO configs
    zero_stage=1,
    decay_lr=True,
    warmup_iters=10000,
    schedule="cosine",
    # System configs
    compile=True,
    model_type="gpt-mha-rope",
)

globals().update(CONFIG.model_dump(exclude_unset=True))
