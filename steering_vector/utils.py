import os
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

def _build_moe_transition(moe_obs_base, moe_obs_steered, num_experts: int) -> np.ndarray:
    T = min(len(moe_obs_base), len(moe_obs_steered))
    M = np.zeros((num_experts, num_experts), dtype=int)
    for t in range(T):
        b = int(moe_obs_base[t]["topk_idx"][0][0])   # base 的 top-1 专家
        s = int(moe_obs_steered[t]["topk_idx"][0][0])# steered 的 top-1 专家
        if 0 <= b < num_experts and 0 <= s < num_experts:
            M[b, s] += 1
    return M

def plot_moe_sankey_plotly(
    moe_obs_base,
    moe_obs_steered,
    num_experts: int,
    layer_idx: int,
    out_html: str = "visualizations/moe_sankey_layer_{layer}.html",
    out_png: str | None = "visualizations/moe_sankey_layer_{layer}.png",
    top_n: int | None = 20,       # 
    min_frac: float = 0.01,      #
    normalize_rows: bool = False  # when true, the plot is a probability distribution of the expert
):
    os.makedirs(os.path.dirname(out_html.format(layer=layer_idx)), exist_ok=True)
    if out_png is not None:
        os.makedirs(os.path.dirname(out_png.format(layer=layer_idx)), exist_ok=True)

    M_full = _build_moe_transition(moe_obs_base, moe_obs_steered, num_experts)
    total = M_full.sum()
    if total == 0:
        print("[Sankey] No flows to plot."); return

    # 行/列裁剪（避免专家过多导致拥挤）
    row_idx = np.arange(num_experts)
    col_idx = np.arange(num_experts)
    M = M_full.copy()
    if (top_n is not None) and (num_experts > top_n):
        r_sum = M.sum(1); c_sum = M.sum(0)
        row_idx = np.argsort(-r_sum)[:top_n]
        col_idx = np.argsort(-c_sum)[:top_n]
        M = M[np.ix_(row_idx, col_idx)]

    # 过滤很小的边（按全局占比）
    flows = []
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v <= 0: 
                continue
            if (v / total) < min_frac:
                continue
            flows.append((i, j, v))
    if not flows:
        print(f"[Sankey] All flows < {min_frac*100:.1f}% of total, nothing to draw."); 
        return

    if normalize_rows:
        row_sums = M.sum(1, keepdims=True).clip(min=1)
        M = M / row_sums
        flows = [(i, j, float(M[i, j])) for (i, j, _) in flows]

    left_labels  = [f"Base {int(r)}" for r in row_idx]
    right_labels = [f"Steered {int(c)}" for c in col_idx]
    labels = left_labels + right_labels

    pal = qualitative.Plotly + qualitative.D3 + qualitative.Set3
    node_colors = [pal[i % len(pal)] for i in range(len(left_labels))] + ["#cccccc"] * len(right_labels)

    sources, targets, values, percents = [], [], [], []
    link_colors = []
    for (i, j, v) in flows:
        sources.append(i)                    
        targets.append(len(left_labels) + j) 
        values.append(v if not normalize_rows else max(v, 1e-9)) 
        frac = (M_full[row_idx[i], col_idx[j]] / total) if not normalize_rows else v
        percents.append(frac)
        link_colors.append(_rgba_with_alpha(node_colors[i], alpha=0.45))

    if not normalize_rows:
        link_hover = "Base %{source.label} → Steered %{target.label}<br>count=%{value}<br>share=%{customdata:.1%}"
    else:
        link_hover = "Base %{source.label} → Steered %{target.label}<br>row-prob=%{customdata:.1%}"

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=12, thickness=16,
            line=dict(color="rgba(0,0,0,0.4)", width=0.5),
            label=labels, color=node_colors
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=link_colors,
            customdata=percents,
            hovertemplate=link_hover
        )
    )])

    subtitle = "(counts & global share)" if not normalize_rows else "(row-normalized prob)"
    fig.update_layout(
        title_text=f"MoE Base→Steered top-1 Sankey (Layer {layer_idx+1}) {subtitle}",
        font_size=11,
        margin=dict(l=10, r=10, t=45, b=10),
        height=600
    )

    # save
    html_path = out_html.format(layer=layer_idx)
    fig.write_html(html_path, include_plotlyjs="cdn")
    if out_png is not None:
        try:
            fig.write_image(out_png.format(layer=layer_idx), scale=2)  # 需要 kaleido
        except Exception as e:
            print(f"[Sankey] PNG export skipped (install kaleido). Error: {e}")
    print(f"[Sankey] Saved: {html_path}")


def _rgba_with_alpha(hex_or_name: str, alpha: float = 0.5) -> str:
    """convert 'rgb(...)'/'#RRGGBB' to 'rgba(r,g,b,a)'."""
    import re
    # Check if it's already in rgb(r,g,b) format
    rgb_match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', hex_or_name)
    if rgb_match:
        r, g, b = rgb_match.groups()
        return f"rgba({r},{g},{b},{alpha:.3f})"
    
    # Otherwise use matplotlib to parse hex colors
    from matplotlib.colors import to_rgba
    r, g, b, _ = to_rgba(hex_or_name)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha:.3f})"
    
def plot_moe_delta_usage(usage_base, usage_steer, layer_idx, outdir="visualizations", top_n=None):
    """
    steered - base
    """
    os.makedirs(outdir, exist_ok=True)
    usage_base = np.asarray(usage_base, dtype=float)
    usage_steer = np.asarray(usage_steer, dtype=float)
    delta = usage_steer - usage_base
    order = np.argsort(-np.abs(delta))
    if top_n is not None:
        order = order[:top_n]

    x = np.arange(len(order))
    plt.figure(figsize=(12, 4))
    plt.bar(x, delta[order], alpha=0.9, edgecolor="black")
    plt.axhline(0, linestyle="--")
    plt.xticks(x, [str(i) for i in order], rotation=0)
    plt.xlabel("Expert ID (sorted by |Δ|)")
    plt.ylabel("Δ usage (steered - base)")
    plt.title(f"MoE Δ usage @ layer {layer_idx+1}")
    plt.tight_layout()
    p = os.path.join(outdir, f"moe_delta_usage_layer_{layer_idx}.png")
    plt.savefig(p, dpi=300)

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

def plot_attention_heatmap(heatmap_data, layer_idx_start):
    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='hot', interpolation='nearest')

    ax.set_xticks(range(heatmap_data.shape[1]))
    ax.set_xticklabels([f'{i}' for i in range(heatmap_data.shape[1])])

    ax.set_yticks(range(len(heatmap_data)))
    ax.set_yticklabels([f'Layer {layer_idx_start + i + 1}' for i in range(len(heatmap_data))])

    ax.set_xlabel('KV Head Index')
    ax.set_ylabel('Layer Index')
    ax.set_title('KV Head Activations Norms')

    plt.colorbar(im, ax=ax, label='Norm')

    ax.set_xticks(np.arange(heatmap_data.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(heatmap_data)) - 0.5, minor=True)

    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f"visualizations/kv_head_activations_heatmap.png", dpi=300)