# LLM-Fine-tune
## Motivation
This repository contains simplified implementations of several prominent **LLM fine-tuning** methods. Developed independently for my research purposes, the code serves as a practical reference for various adaptation techniques.

This repo is built based on [sotopia-rl](https://github.com/sotopia-lab/sotopia-rl). If you find this useful, don't forget to star that repo too.

The collection continues to expand as my research demands new methodologies.

# Environment Setup
Recommand to use **conda** to manage environment
```
conda create -n myenv python==3.10
conda activate myenv
pip install -r requirements.txt
```

## Current method
1. [SFT](#detailed-infor-about-sft)

Use ```bash run.sh``` to run all the methods.

## Load dataset

I provide some example of loading datasets. If you want to load your customed datasets, add a function in [load_dataset.py](./utils/data/load_dataset.py).

## Detailed infor about SFT

1. SFT setting code is in [sft.sh](./train/sft/sft.py)
2. SFT training code is in [sft_trainer.py](./utils/trainer/sft_trainer.py)
3. SFT data format should be
```
[
    {
        "input":<your input>,
        "output":<your output>
    },
    {
        "input":<your input>,
        "output":<your output>
    },
    ...
]
```
