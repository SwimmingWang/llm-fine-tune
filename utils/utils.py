from datasets import Dataset
from datasets import load_dataset
import random
import json
def load_dataset_mc_sft(cache_dir: str):
    '''
    load Multifaceted-Collection dataset for sft
    '''
    ds = load_dataset("kaist-ai/Multifaceted-Collection", split="train",cache_dir=cache_dir)
    ds = ds.select(range(1000))
    train_dataset = process_to_sft(ds)

    return train_dataset

def load_dataset_mc_dpo(cache_dir: str, train_size: int, val_size: int):
    
    ds = load_dataset("kaist-ai/Multifaceted-Collection-DPO", split="train", cache_dir=cache_dir)

    # 随机打乱数据集
    ds = ds.shuffle(seed=42)
    
    train_dataset = []
    val_dataset = []
    
    train_dataset = process_to_dpo(ds.select(range(train_size)))
    val_dataset = process_to_dpo(ds.select(range(train_size, train_size + val_size)))
    
    return train_dataset, val_dataset

def process_to_sft(ds):
    def merge_fields(example):
        # 合并 system 和 prompt
        if 'system' in example:
            prompt = example['system'].strip() + '\\n' + example['prompt'].strip()
        else:
            prompt = example['prompt'].strip()
        return {
            'prompt': prompt,
            'response': example['output']
        }
    return ds.map(merge_fields)

def process_to_dpo(ds):
    def merge_fields(example):
        # 合并 system 和 prompt
        if 'system' in example and example['system']:
            prompt = example['system'].strip() + '\\n' + example['prompt'].strip()
        else:
            prompt = example['prompt'].strip()
        return {
            'prompt': prompt,
            'chosen': example['chosen'],
            'rejected': example['rejected']
        }
    return ds.map(merge_fields)