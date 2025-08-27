from tqdm import tqdm
import json
import torch
import torch.nn.functional as F
import os
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, default="prompt", choices=["prompt", "response"])
args = parser.parse_args()

target = args.target

activation_dir = "./activations"
pos_tensors = defaultdict(list)
neg_tensors = defaultdict(list)

for file in os.listdir(activation_dir):
    if file.endswith(".pt") and target in file:
        file_path = os.path.join(activation_dir, file)
        category = file.split("_")[2]
        cur_tensor = torch.load(file_path)
        if "pos" in file:
            pos_tensors[category].append(cur_tensor) # category: list of n_tensors (n_layer, batch, hidden)
        elif "neg" in file:
            neg_tensors[category].append(cur_tensor)

min_n_tensors = min(len(v) for v in list(pos_tensors.values()) + list(neg_tensors.values()))
pos_tensors = [torch.stack(v[:min_n_tensors], dim=0) for v in pos_tensors.values()] # list of n_categories (n_tensors, n_layer, batch, hidden)
neg_tensors = [torch.stack(v[:min_n_tensors], dim=0) for v in neg_tensors.values()]

pos_tensors = torch.stack(pos_tensors, dim=0) # (n_categories, n_tensors, n_layer, batch, hidden)
neg_tensors = torch.stack(neg_tensors, dim=0)

print(f"n_categories x n_tensors x n_layers x batch (1) x hidden_dim: {pos_tensors.shape}")

pos_tensors = pos_tensors.squeeze(-2) # (n_categories, n_tensors, n_layer, hidden)
neg_tensors = neg_tensors.squeeze(-2)

num_layers = pos_tensors.shape[-2]

# Compute the difference between the negative and positive prompt tensors to get the evil vector diff.
diff_vectors = []
for l in range(num_layers):
    pos_layer = pos_tensors[:, :, l, :] # (n_categories, n_tensors, hidden)
    neg_layer = neg_tensors[:, :, l, :]

    pos_mean = pos_layer.mean(dim=1) # (n_categories, hidden)
    neg_mean = neg_layer.mean(dim=1)

    diff_mean = neg_mean - pos_mean # category wise diff

    diff_layer_norm = F.normalize(diff_mean, dim=-1)

    diff_vector = diff_layer_norm.mean(dim=0) # (hidden,)

    diff_vectors.append(diff_vector)

diff_vectors = torch.stack(diff_vectors, dim=0) # (n_layer, hidden)
print(f"Diff vector shape: {diff_vectors.shape}")

torch.save(diff_vectors, "activations/diff_vectors.pt")