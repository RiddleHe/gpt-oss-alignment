from utils import plot_attention_heatmap
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TopPLogitsWarper, TemperatureLogitsWarper,
)
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--layer_idx_start", type=int, default=0)
args = parser.parse_args()

top_k = args.top_k
layer_idx_start = args.layer_idx_start

model_id = "openai/gpt-oss-20b"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

steering_vectors = torch.load("activations/diff_vectors.pt") # (n_layers, hidden)
steering_vectors = steering_vectors.to(model.device).to(torch.bfloat16) # (n_layers, hidden)
print(f"Steering vectors shape: {steering_vectors.shape}")

hidden_steering_vectors = steering_vectors[1:, :] # discard embedding layer
chosen_steering_vectors = hidden_steering_vectors[layer_idx_start:, :]

logit_head = model.lm_head # use lm head, not embedding
logits = logit_head(chosen_steering_vectors)

top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)

# Convert top_k_indices to list of tokens
top_k_indices = top_k_indices.tolist()
for i, layer in enumerate(top_k_indices):
    layer_tokens = [tokenizer.decode(idx) for idx in layer]
    print(f"Layer {layer_idx_start+i}: {layer_tokens}")

kv_heads_activations = []
num_kv_heads = model.config.num_key_value_heads

print(f"Number of model layers: {len(model.model.layers)}")
print(f"Number of steering vectors: {len(hidden_steering_vectors)}")
assert len(model.model.layers) == len(hidden_steering_vectors), f"Number of model layers does not match number of vectors"
for layer_idx, decoder_layer in tqdm(enumerate(model.model.layers), desc="Computing kv head norms"):
    attn_layer = decoder_layer.self_attn

    if layer_idx < layer_idx_start:
        continue

    vector = hidden_steering_vectors[layer_idx]
    k_steering = F.linear(vector, attn_layer.k_proj.weight, attn_layer.k_proj.bias)
    k_steering = k_steering.view(num_kv_heads, attn_layer.head_dim)
    k_norm_per_head = k_steering.norm(dim=-1)

    k_norm_per_head = k_norm_per_head.float().cpu().tolist()
    kv_heads_activations.append(k_norm_per_head)

print(f"kv head activations: {len(kv_heads_activations)} x {len(kv_heads_activations[0])}")

heatmap_data = np.array(kv_heads_activations)
plot_attention_heatmap(heatmap_data, layer_idx_start)