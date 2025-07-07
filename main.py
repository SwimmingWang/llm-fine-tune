#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) LoRA微调代码
使用Transformers和PEFT库进行微调
"""

import os
import torch
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    pipeline,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from config.config import Config
import wandb
import argparse
import shutil

def setup_model_and_tokenizer(config):
    """设置模型和tokenizer"""
    # 量化配置 (可选)
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=True,
    # )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        # quantization_config=bnb_config,  # 注释这行如果不使用量化
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # LoRA配置
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def main(args):
    torch.manual_seed(Config.seed)

    # 解析参数
    do_sft = args.do_sft
    do_dpo = args.do_dpo
    sft_model_dir = args.sft_model_dir
    dpo_model_dir = args.dpo_model_dir
    load_dataset_type = args.load_dataset_type

    # Load model
    model, tokenizer = setup_model_and_tokenizer(Config)

    # SFT流程
    # if do_sft:
    #     print("Loading SFT dataset...")
        
    #     if load_dataset_type == "mutilfaceted_collection":
    #         from utils.utils import load_dataset_mc_sft
    #         train_dataset = load_dataset_mc_sft(cache_dir=args.cache_dir)
    #     else:
    #         pass

    #     from train.sft import train_sft
    #     print("Training SFT...")
    #     sft_trainer = train_sft(model, tokenizer, train_dataset)        
    #     print("SFT done!")
    
    # DPO流程
    if do_dpo:
        # 预处理数据

        if load_dataset_type == "mutilfaceted_collection":
            from utils.utils import load_dataset_mc_dpo
            train_dataset, val_dataset = load_dataset_mc_dpo(cache_dir=args.cache_dir, train_size=args.train_size, val_size=args.val_size)
        else:
            pass
        from train.dpo import train_dpo
        print("Training DPO...")
        dpo_trainer = train_dpo(train_dataset, val_dataset, model, tokenizer)
        print("DPO done!")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="/common/users/sw1525/Qwen2.5-0.5B-Instruct", help="model name")
    args.add_argument("--train_size", type=int, default=8000, help="train size")
    args.add_argument("--val_size", type=int, default=1000, help="validation size")
    args.add_argument("--seed", type=int, default=42, help="random seed")
    args.add_argument("--use_wandb", type=bool, default=True, help="use wandb")
    args.add_argument("--do_sft", type=bool, default=True, help="do sft training")
    args.add_argument("--do_dpo", type=bool, default=True, help="do dpo training")
    args.add_argument("--sft_model_dir", type=str, default=None, help="sft model directory, loaded when dpo")
    args.add_argument("--dpo_model_dir", type=str, default=None, help="dpo model directory, loaded when sft")
    args.add_argument("--load_dataset_type", type=str, default="mutilfaceted_collection", help="load dataset type")
    args.add_argument("--cache_dir", type=str, default="", help="cache directory")
    args = args.parse_args()
    main(args)