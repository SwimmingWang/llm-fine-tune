export CUDA_VISIBLE_DEVICES=4,5,6,7
export N_GPUS=4
# echo "Loading dataset..."
# python utils/load_dataset.py \
#     --cache_dir /common/home/sw1525/lora/cache \
#     --load_dataset_name mutilfaceted_collection \
#     --load_dataset_type sft 

echo "Training..."
python main.py \
    --model_name /common/users/sw1525/Qwen2.5-0.5B-Instruct \
    --train_size 8000 \
    --val_size 1000 \
    --seed 42 \
    --use_wandb True \
    --do_sft False \
    --do_dpo True \
    --sft_model_dir /common/home/sw1525/lora/sft_lora_output \
    --dpo_model_dir /common/home/sw1525/lora/dpo_lora_output 