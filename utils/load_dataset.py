from datasets import load_dataset
import argparse

def load_dataset_mc_sft(cache_dir: str, load_dataset_name: str):
    '''
    load Multifaceted-Collection dataset for sft
    '''
    if load_dataset_name == "mutilfaceted_collection":
        print("Loading Multifaceted-Collection dataset for sft...")
        if cache_dir == "":
            ds = load_dataset("kaist-ai/Multifaceted-Collection", split="train")
        else:
            ds = load_dataset("kaist-ai/Multifaceted-Collection", split="train",cache_dir=cache_dir)
    else:
        raise ValueError(f"Invalid dataset name: {load_dataset_name}")

def load_dataset_mc_dpo(cache_dir: str, load_dataset_name: str):
    """
    load dpo dataset
    
    the dataset should contain the following fields:
    - prompt: user input/question
    - chosen: preferred answer
    - rejected: rejected answer
    
    return: Dataset object, containing train and validation split
    """
    if load_dataset_name == "mutilfaceted_collection":
        print("Loading Multifaceted-Collection dataset for dpo...")
        if cache_dir == "":
            ds = load_dataset("kaist-ai/Multifaceted-Collection-DPO", split="train")
        else:
            ds = load_dataset("kaist-ai/Multifaceted-Collection-DPO", split="train",cache_dir=cache_dir)
    else:
        raise ValueError(f"Invalid dataset name: {load_dataset_name}")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--load_dataset_name", type=str, default="mutilfaceted_collection")
    args.add_argument("--load_dataset_type", type=str, default="dpo")
    args.add_argument("--cache_dir", type=str, default="")
    args = args.parse_args()
    if args.load_dataset_type == "dpo":
        load_dataset_mc_dpo(cache_dir=args.cache_dir, load_dataset_name=args.load_dataset_name)
    elif args.load_dataset_type == "sft":
        load_dataset_mc_sft(cache_dir=args.cache_dir, load_dataset_name=args.load_dataset_name)
    else:
        raise ValueError(f"Invalid dataset type: {args.load_dataset_type}")