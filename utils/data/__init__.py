# from .Alpaca_CoT import load_alpaca_cot_dataset
from .data import SFTDataset

def load_data(data_name: str, split: str):
    if data_name == "alpaca_cot":
        return load_alpaca_cot_dataset(split)
    else:
        raise ValueError(f"Unsupported data name: {data_name}")