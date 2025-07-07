class Config:
    # 模型相关
    model_name = "/common/users/sw1525/Qwen2.5-0.5B-Instruct"  # 替换为你的基础模型
    model_max_length = 1024
    
    # LoRA参数
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    target_modules = ["q_proj", "v_proj"]  # 根据模型调整
    
    # 训练参数
    output_dir = "/common/home/sw1525/lora/dpo_lora_output"
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    gradient_accumulation_steps = 2
    learning_rate = 1e-4
    num_train_epochs = 1
    warmup_steps = 100
    logging_steps = 10
    eval_steps = 100
    save_steps = 500
    
    # DPO参数
    beta = 0.1  # DPO loss的温度参数
    max_length = 512
    max_prompt_length = 256
    
    # 其他
    seed = 42
    bf16 = True
    gradient_checkpointing = False
    use_wandb = True  # 设置为True以使用wandb
    sft_output_dir = "/common/home/sw1525/lora/model/sft_lora_output"
    dpo_output_dir = "/common/home/sw1525/lora/model/dpo_lora_output"