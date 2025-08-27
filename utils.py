import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
from typing import Dict, Any, List, Optional, Tuple

def compute_moe_stats(
    moe_obs_base: List[dict],
    moe_obs_steered: List[dict],
    num_experts: Optional[int],
    topk_key: str = "topk_idx"
) -> Dict[str, Any]:
    """Compute expert usage, top-1 switches, and Jaccard distances per step.
    Returns a dict with arrays/lists; no tokenizer/IO dependencies.
    """
    if not num_experts:
        return {"num_experts": 0, "steps": 0, "usage_base": [], "usage_steer": [],
                "switch_top1": [], "jaccard": [], "most_switch_step": -1}

    E = int(num_experts)
    T = min(len(moe_obs_base), len(moe_obs_steered))
    if T == 0:
        return {"num_experts": E, "steps": 0, "usage_base": [0]*E, "usage_steer": [0]*E,
                "switch_top1": [], "jaccard": [], "most_switch_step": -1}

    usage_base = np.zeros(E, dtype=int)
    usage_steer = np.zeros(E, dtype=int)
    switch_top1: List[int] = []
    jaccard: List[float] = []

    for t in range(T):
        b_idx = [int(i) for i in moe_obs_base[t][topk_key][0]]
        s_idx = [int(i) for i in moe_obs_steered[t][topk_key][0]]

        usage_base[b_idx] += 1
        usage_steer[s_idx] += 1

        switch_top1.append(1 if b_idx[0] != s_idx[0] else 0)
        b_set, s_set = set(b_idx), set(s_idx)
        inter = len(b_set & s_set)
        union = len(b_set | s_set)
        jaccard.append(1.0 - (inter / union if union > 0 else 1.0))

    # choose the step with largest Jaccard; nudge ties by top-1 switch
    most_switch_step = int(np.argmax(np.array(jaccard) + 1e-3*np.array(switch_top1)))

    return {
        "num_experts": E,
        "steps": T,
        "usage_base": usage_base.tolist(),
        "usage_steer": usage_steer.tolist(),
        "switch_top1": switch_top1,
        "jaccard": jaccard,
        "most_switch_step": most_switch_step,
    }

def decode_step_token(
    generated_ids, prompt_len: int, step: int
) -> Optional[int]:
    """Helper that maps a decoding step t -> token_id at position prompt_len + t."""
    pos = prompt_len + step
    if pos < generated_ids.size(1):
        return int(generated_ids[0, pos].item())
    return None

def plot_moe_usage_hist(usage_base, usage_steer, layer_idx, outdir="visualizations"):
    os.makedirs(outdir, exist_ok=True)
    E = len(usage_base)
    x = np.arange(E)
    w = 0.45
    plt.figure(figsize=(12,4))
    plt.bar(x - w/2, usage_base, width=w, label="base", alpha=0.85, edgecolor="black")
    plt.bar(x + w/2, usage_steer, width=w, label="steered", alpha=0.85, edgecolor="black")
    plt.xlabel("Expert ID"); plt.ylabel("Top-k hits")
    plt.title(f"MoE expert usage @ layer {layer_idx+1}")
    plt.legend(); plt.grid(axis="y", alpha=0.25)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    p = os.path.join(outdir, f"moe_usage_layer_{layer_idx}.png")
    plt.savefig(p, dpi=300)

def plot_moe_switch_curves(switch_top1, jaccard, layer_idx, outdir="visualizations"):
    os.makedirs(outdir, exist_ok=True)
    T = len(switch_top1); steps = np.arange(T)
    cum_rate = np.cumsum(switch_top1) / np.maximum(1, steps+1)

    fig, ax = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    ax[0].bar(steps, switch_top1, alpha=0.85)
    ax[0].set_ylabel("Top-1 switch (0/1)")
    ax[0].set_title(f"MoE switches @ layer {layer_idx+1}")
    ax[0].grid(alpha=0.25, axis="y")
    ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))

    ax[1].plot(steps, jaccard, marker="o", ms=3, label="Top-k Jaccard distance")
    ax[1].plot(steps, cum_rate, marker="s", ms=3, label="Cumulative top-1 switch rate")
    ax[1].set_xlabel("Decoding step"); ax[1].set_ylabel("Score")
    ax[1].grid(alpha=0.3); ax[1].legend()
    plt.tight_layout()
    p = os.path.join(outdir, f"moe_switches_layer_{layer_idx}.png")
    plt.savefig(p, dpi=300)

def plot_moe_topk_heatmap(moe_obs_base, moe_obs_steered, num_experts, layer_idx, top_k=2, outdir="visualizations"):
    """
    2-row heatmap: row0=base, row1=steered. Each cell shows the top-1 expert ID at that step.
    (If you want top-k, plot multiple rows or annotate.)
    """
    os.makedirs(outdir, exist_ok=True)
    T = min(len(moe_obs_base), len(moe_obs_steered))
    y = np.zeros((2, T), dtype=int)
    for t in range(T):
        y[0, t] = int(moe_obs_base[t]["topk_idx"][0][0])
        y[1, t] = int(moe_obs_steered[t]["topk_idx"][0][0])

    plt.figure(figsize=(12, 2.8))
    im = plt.imshow(y, aspect="auto", interpolation="nearest", vmin=0, vmax=max(1, num_experts-1))
    plt.yticks([0,1], ["base","steered"])
    plt.xlabel("Decoding step"); plt.title(f"Top-1 expert timeline @ layer {layer_idx+1}")
    cbar = plt.colorbar(im, shrink=0.8); cbar.set_label("Expert ID")
    plt.tight_layout()
    p = os.path.join(outdir, f"moe_top1_heatmap_layer_{layer_idx}.png")
    plt.savefig(p, dpi=300)

def plot_attention_diff(attn_base, attn_steered, token_labels, layer_idx=19, step=0):
    os.makedirs("visualizations", exist_ok=True)
    
    attn_base = attn_base.float().numpy()
    attn_steered = attn_steered.float().numpy()

    x_positions = range(len(token_labels))
    plt.plot(x_positions, attn_base, label="base", alpha=0.7, marker="o", markersize=4)
    plt.plot(x_positions, attn_steered, label="steered", alpha=0.7, marker="o", markersize=4)

    diff = attn_steered - attn_base
    threshold = np.std(diff)
    threshold_attn = np.max(attn_base) * 0.1

    important_idx = np.where(
        (attn_base > threshold_attn) |
        (attn_steered > threshold_attn) |
        (np.abs(diff) > threshold)
    )[0]

    plt.figure(figsize=(14, 5))
    plt.plot(attn_base, label="Base", alpha=0.7, marker="o", markersize=3)
    plt.plot(attn_steered, label="Steered", alpha=0.7, marker="s", markersize=3)

    for idx in important_idx:
        plt.annotate(
            token_labels[idx],
            xy=(idx, max(attn_base[idx], attn_steered[idx])),
            xytext=(0, 5), textcoords="offset points",
            rotation=45, fontsize=7, ha="left"
        )
    plt.xlabel("Token position")
    plt.ylabel("Attention weight")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"Attention pattern at layer {layer_idx + 1} at decoding step {step}")
    plt.savefig(f"visualizations/attention_layer_{layer_idx}_step_{step}.png", dpi=300, bbox_inches="tight")

def plot_attention_heads(norm_per_head, mean_norm, std_norm, layer_idx, step, norm_type="Q-norm"):
    plt.figure(figsize=(8, 5))
    plt.hist(norm_per_head.float().cpu().numpy(), bins=20, alpha=0.7, edgecolor="black")
    plt.axvline(x=mean_norm, color='r', linestyle='--', label=f"Mean: {mean_norm:.0f}")
    plt.axvline(x=mean_norm + std_norm, color="blue", linestyle=":", label=f"+1σ: {mean_norm + std_norm:.0f}")
    plt.axvline(x=mean_norm + 2*std_norm, color="orange", linestyle=":", label=f"+2σ: {mean_norm + 2*std_norm:.0f}")
    plt.xlabel(f"{norm_type} of steering vector")
    plt.ylabel("Number of heads")
    plt.title(f"Distribution of {norm_type}s of steering vector across attention heads at layer {layer_idx + 1}")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.savefig(f"visualizations/attention_head_{norm_type}s_layer_{layer_idx}_step_{step}.png", dpi=300)