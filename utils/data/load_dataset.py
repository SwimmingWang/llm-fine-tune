import argparse
from datasets import load_dataset
import json

def load_kaist_ai_multifaceted_collection(split: str, cache_dir: str, output_path: str, num_samples: int):
    dataset = load_dataset("kaist-ai/Multifaceted-Collection", split=split, cache_dir=cache_dir)
    if num_samples != -1:
        dataset = dataset.select(range(num_samples))
    
    dataset = dataset.to_dict()
    new_data = []
    for i in range(num_samples if num_samples != -1 else len(dataset["system"])):
        item = {
            "input": dataset["system"][i] + dataset["prompt"][i],
            "output": dataset["output"][i]
        }
        new_data.append(item)
    # save to json file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def main(args):
    if args.dataset_name == "kaist-ai/Multifaceted-Collection":
        if args.split == "train":
            dataset = load_kaist_ai_multifaceted_collection(args.split, args.cache_dir, args.output_path, args.num_samples)
        else:
            raise ValueError(f"Split {args.split} not found")
    else:
        raise ValueError(f"Dataset {args.dataset_name} not found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    args = parser.parse_args()
    print(f"Loading dataset: {args.dataset_name} {args.split}")
    main(args)
    print(f"Successfully saved dataset {args.dataset_name} {args.split} to {args.output_path}")
    