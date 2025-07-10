from datasets import load_dataset


def load_alpaca_cot_dataset(cache_dir: str, split: str):
    ds = load_dataset("QingyiSi/Alpaca-CoT", split, cache_dir=cache_dir)
    return ds