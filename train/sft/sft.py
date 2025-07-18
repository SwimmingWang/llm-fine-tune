import argparse
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from accelerate import Accelerator

from utils.trainer.sft_trainer import SFTTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reward model using SFT with LoRA.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=2, help="Validation batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--sft_data_path", type=str, required=True, help="Path to SFT data")
    parser.add_argument("--template_path", type=str, required=True, help="Path to the Jinja template file")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--evaluation_steps", type=int, default=100, help="Evaluation interval in steps")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    # LoRA-specific arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, default="c_attn,q_proj,v_proj", help="Target modules for LoRA")
    # Checkpoint and Wandb arguments
    parser.add_argument("--checkpoint_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--lora_checkpoint_path", type=str, default=None, help="Path to load LoRA checkpoint")
    parser.add_argument("--wandb_project", type=str, default="sft-project", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="sft-run", help="Wandb run name")
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA (4-bit) for model loading.")

    args = parser.parse_args()
    accelerator = Accelerator()

    trainer = SFTTrainer(args, accelerator)
    trainer.train()