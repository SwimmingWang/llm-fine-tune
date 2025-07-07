from trl import SFTTrainer, SFTConfig
from config.config import Config
def sft_preprocess_function(examples, tokenizer):
    # 支持 batch，输入是 {"prompt": [...], "response": [...]}
    texts = [f"{p} {r}" for p, r in zip(examples["prompt"], examples["response"])]
    
    # 先获取每个 prompt 的 token 数，计算 cutoff 位置
    prompt_lens = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in examples["prompt"]]

    # 编码整句
    model_inputs = tokenizer(
        texts,
        max_length=Config.model_max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    labels = input_ids.clone()
    for i, prompt_len in enumerate(prompt_lens):
        labels[i, :prompt_len] = -100  # mask掉prompt部分

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def train_sft(model, tokenizer, train_dataset):
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    train_dataset = train_dataset.map(
        lambda x: sft_preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    training_args = SFTConfig(
        output_dir=Config.sft_output_dir,
        per_device_train_batch_size=Config.per_device_train_batch_size,
        per_device_eval_batch_size=Config.per_device_eval_batch_size,
        gradient_accumulation_steps=Config.gradient_accumulation_steps,
        learning_rate=Config.learning_rate,
        num_train_epochs=Config.num_train_epochs,
        warmup_steps=Config.warmup_steps,
        logging_steps=Config.logging_steps,
        eval_steps=Config.eval_steps,
        save_steps=Config.save_steps,
        save_strategy="no",
        # evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=Config.bf16,
        gradient_checkpointing=Config.gradient_checkpointing,
        dataloader_drop_last=True,
        report_to="wandb" if Config.use_wandb else None,
        run_name="sft-lora-training",
        seed=Config.seed,
    )
    sft_trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    print("Starting SFT training...")
    sft_trainer.train()
    print("Saving SFT model...")
    sft_trainer.save_model()
    tokenizer.save_pretrained(Config.sft_output_dir)
    return sft_trainer
