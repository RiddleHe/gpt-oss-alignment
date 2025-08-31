import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Iterator
import torch
import torch.nn as nn
import torch.optim as optim
import sys

from model import Autoencoder, TopK
from loss import normalized_mean_squared_error

@dataclass
class TrainingConfig:
    n_latents: int = 8192
    d_model: int = 4096
    k: int = 32

    batch_size: int = 1024
    lr: float = 1e-4
    max_steps: int = 100000

    data_dir: str = ""
    max_samples: int | None = None

    checkpoint_every: int = 1000
    output_dir: str = "./checkpoints"

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            return cls(**yaml.safe_load(f))

def load_activation_data(config):
    data_path = Path(config.data_dir)
    data = torch.load(data_path)

    if config.max_samples is not None:
        data = data[:config.max_samples]

    return data.float()

def create_data_iterator(config):
    data = load_activation_data(config)
    
    dataset_size = len(data)
    idx = 0
    while True:
        batch_end = min(idx + config.batch_size, dataset_size)
        batch = data[idx:batch_end]
        yield batch

        idx = batch_end
        if idx >= dataset_size:
            idx = 0
    
def create_model(config):
    topk_activation = TopK(k=config.k, postact_fn=nn.ReLU())
    
    model = Autoencoder(
        n_latents=config.n_latents,
        n_inputs=config.d_model,
        activation=topk_activation,
        tied=False,
        normalize=True,
    ).cuda()

    return model

def train_step(model, batch, optimizer):
    optimizer.zero_grad()

    latents_pre_act, latents, reconstruction = model(batch)
    loss = normalized_mean_squared_error(reconstruction, batch)

    loss.backward()
    optimizer.step()

    return {
        'loss': loss.item(),
        'sparsity': (latents > 0).float().mean().item()
    }

def train_model(config):
    model = create_model(config)
    data_iter = create_data_iterator(config)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    for step in range(config.max_steps):
        batch = next(data_iter).cuda()

        metrics = train_step(model, batch, optimizer)
        print(f"Step {step}: Loss={metrics['loss']:.6f}, Sparsity={metrics['sparsity']:.6f}")

    return model

def sample(reference_model, sae_model, text_samples, layer_idx, device="cuda"):
    pass

def main():
    config_path = sys.argv[1]

    config = TrainingConfig.from_yaml(config_path)
    print(f"Loaded config from {config_path}")

    print(f"Starting training...")
    model = train_model(config)
    print(f"Training complete")

    if config.output_dir:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k not in ['activation.k', 'activation.postact_fn']}
        torch.save(state_dict, f"{config.output_dir}/model.pt")
        print(f"Saved model to {config.output_dir}")

if __name__ == "__main__":
    main()