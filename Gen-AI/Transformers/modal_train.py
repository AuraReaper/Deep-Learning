import modal
from pathlib import Path
import torch
import os
import time

# Create a Modal image with all required dependencies
image = modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04" , add_python="3.12").pip_install(
    "torch",
    "tokenizers",
    "datasets",
    "torchmetrics",
    "tensorboard",
    "tqdm"
)

image.add_local_dir("." , remote_path="/root")

from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config
from train import train_model

app = modal.App("transformer-training" , image=image)

@app.function(
    image=image,
    gpu="H100",  # Using H100
    timeout=36000,

)
def train_transformer():
    start_time = time.time()

    #Change to the mounted directory
    os.chdir("/root/")

    # Import your modules

    # Get the configuration
    config = get_config()

    # Optimize config for H100 GPU
    config.update({
        "batch_size": 16,# Large batch size for H100
        "num_epochs": 5,
        "lr": 3e-4,  # Increased learning rate for larger batch size
        "experiment_name": "/root/transformer/runs/tmodel",
        "model_folder": "/root/transformer/weights",
        "tokenizer_file": "/root/transformer/tokenizer_{0}.json",
        # Increased model size to utilize H100's compute power
        "d_model": 512,
        "d_ff": 3072,
        "num_layers": 6,
        "num_heads": 12,
        "dropout": 0.1
    })

    # Create necessary directories
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    Path(config['experiment_name']).mkdir(parents=True, exist_ok=True)

    # Print GPU information
    if torch.cuda.is_available():
        print("\n=== Hardware Information ===")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print("===========================\n")

    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Start training
    print("Starting training on Modal H100...")
    train_model(config)

    # Calculate total time and estimated cost
    total_time = (time.time() - start_time) / 3600  # Convert to hours
    estimated_cost = total_time * 3.80  # H100 costs approximately $3.80 per hour

    print(f"\n=== Training Summary ===")
    print(f"Total training time: {total_time:.2f} hours")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    print(f"Remaining credits (approximate): ${30 - estimated_cost:.2f}")
    print("=====================\n")

    return config['model_folder']


@app.local_entrypoint()
def main():
    print("Initializing transformer training on Modal H100 GPU...")
    try:
        weights_path = train_transformer.remote()
        print(f"Training completed successfully!")
        print(f"Model weights saved at: {weights_path}")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
    finally:
        print("\nDon't forget to check your remaining credits at modal.com/usage")
