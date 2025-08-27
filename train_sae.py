"""
This script is in development, not ready for training.
"""

import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    StandardTrainingSAEConfig,
    LoggingConfig,
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="openai/gpt-oss-20b")
parser.add_argument("--dataset_path", typ=str, default="")
args = parser.parse_args()

model_name_or_path = args.model_name_or_path
dataset_path = args.dataset_path

# Define total training steps and batch size
total_training_steps = 30_000
batch_size = 4096
total_training_tokens = total_training_steps * batch_size

# Learning rate and L1 warmup schedules
lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

device = "cuda"

cfg = LanguageModelSAERunnerConfig(
    model_class_name="AutoModelForCausalLM",
    model_name=model_name_or_path,
    hook_name="", # TODO: add hook name
    dataset_path=dataset_path,
    streaming=True,

    # SAE Parameters are in the nested 'sae' config
    sae=StandardTrainingSAEConfig(
        d_in=1024, 
        d_sae=16 * 1024,
        apply_b_dec_to_input=True,
        normalize_activations="expected_average_only_in",
        l1_coefficient=5,
        l1_warm_up_steps=l1_warm_up_steps,
    ),

    # Training Parameters
    lr=5e-5,
    lr_warm_up_steps=lr_warm_up_steps,
    lr_decay_steps=lr_decay_steps,
    train_batch_size_tokens=batch_size,

    # Activation Store Parameters
    context_size=256,
    n_batches_in_buffer=64,
    training_tokens=total_training_tokens,
    store_batch_size_prompts=16,

    # WANDB
    logger=LoggingConfig(
        log_to_wandb=True,
        wandb_project="sae_lens",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
    ),

    # Misc
    device=device,
    seed=42,
    n_checkpoints=4,
    checkpoint_path="checkpoints",
    dtype="float32"
)
sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()