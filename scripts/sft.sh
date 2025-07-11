export MODEL_PATH="/common/users/sw1525/Qwen2.5-3B-Instruct"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file config/accelerate_config_sft.yaml \
  --main_process_port 29512 \
    train/sft/sft.py \
    --model_name $MODEL_PATH \
    --learning_rate 1e-4 \
    --max_length 4096 \
    --train_batch_size 2 \
    --val_batch_size 1 \
    --accumulation_steps 8 \
    --num_epochs 500 \
    --use_lora \
    --evaluation_steps 5 \
    --sft_data_path /common/users/sw1525/data/kaist-ai/Multifaceted-Collection/train.json \
    --template_path utils/tools/qwen.jinja \
    --checkpoint_dir models/sft_checkpoints_qwen2.5-3b \