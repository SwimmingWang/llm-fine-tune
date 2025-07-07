from config.config import Config
from trl import DPOTrainer, DPOConfig
"""预处理数据集"""
def dpo_preprocess_function(examples, tokenizer):
        def tokenize_text(text):
            return tokenizer(
                text,
                truncation=True,
                max_length=Config.model_max_length,
                padding="max_length",
                return_tensors="pt"
            )
        
        # 处理prompt, chosen, rejected
        prompts = examples["prompt"]
        chosen = examples["chosen"]
        rejected = examples["rejected"]
        
        # 组合prompt和response
        chosen_texts = [f"{p} {c}" for p, c in zip(prompts, chosen)]
        rejected_texts = [f"{p} {r}" for p, r in zip(prompts, rejected)]
        
        # Tokenize
        chosen_tokens = tokenize_text(chosen_texts)
        rejected_tokens = tokenize_text(rejected_texts)
        
        return {
            "prompt": prompts,
            "chosen": chosen_texts,
            "rejected": rejected_texts,
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
        }
    
def train_dpo(train_dataset, val_dataset, model, tokenizer):
    print("Preprocessing dataset...")
    train_dataset = train_dataset.map(
        lambda x: dpo_preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: dpo_preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    # DPO训练参数
    training_args = DPOConfig(
        beta=Config.beta,
        max_prompt_length=Config.max_prompt_length,
        max_length=Config.max_length,
        output_dir=Config.dpo_output_dir,
        per_device_train_batch_size=Config.per_device_train_batch_size,
        per_device_eval_batch_size=Config.per_device_eval_batch_size,
        gradient_accumulation_steps=Config.gradient_accumulation_steps,
        learning_rate=Config.learning_rate,
        num_train_epochs=Config.num_train_epochs,
        warmup_steps=Config.warmup_steps,
        logging_steps=Config.logging_steps,
        eval_steps=Config.eval_steps,
        save_steps=Config.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=Config.bf16,
        gradient_checkpointing=Config.gradient_checkpointing,
        dataloader_drop_last=True,
        report_to="wandb" if Config.use_wandb else None,
        run_name="dpo-lora-training",
        seed=Config.seed,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    print("Starting DPO training...")
    trainer.train()
    print("Saving DPO model...")
    trainer.save_model()
    tokenizer.save_pretrained(Config.dpo_output_dir)
    print("All done!")