import torch
import torch.nn.functional as F
import torch.nn as nn


def create_steering_hook(vector, current_strength):
    def hook(module, input, output):
        hidden = output[0]
        if hidden.dim() == 2:
            hidden[-1, :] += current_strength * vector
        elif hidden.dim() == 3:
            hidden[:, -1, :] += current_strength * vector
        return output
    return hook


def capture_attn_hook(storage_list, prompt_len):
    def attn_hook(module, input, output):
        _, attn_weights = output
        attn = attn_weights[0, :, -1, :prompt_len].cpu()
        storage_list.append(attn)
    return attn_hook


def capture_attn_head_hook(past_kv, strength, vector, BEST_LAYER, visualize_step, plot_attention_heads):
    def hook(module, input, output):
        steering_hidden = (vector * strength).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)

        q_steering = F.linear(steering_hidden, module.q_proj.weight, module.q_proj.bias)
        k_steering = F.linear(steering_hidden, module.k_proj.weight, module.k_proj.bias)
        v_steering = F.linear(steering_hidden, module.v_proj.weight, module.v_proj.bias)

        layer_idx = BEST_LAYER
        k_cached = past_kv[layer_idx][0]
        v_cached = past_kv[layer_idx][1]

        head_dim = module.head_dim
        num_kv_heads = k_cached.shape[1]
        num_heads = num_kv_heads * module.num_key_value_groups

        q_steering = q_steering.view(1, 1, num_heads, head_dim).transpose(1, 2)
        k_steering = k_steering.view(1, 1, num_kv_heads, head_dim).transpose(1, 2)
        v_steering = v_steering.view(1, 1, num_kv_heads, head_dim).transpose(1, 2)

        q_norm_per_head = q_steering.norm(dim=-1).squeeze()
        k_norm_per_head = k_steering.norm(dim=-1).squeeze()
        v_norm_per_head = v_steering.norm(dim=-1).squeeze()

        mean_q_norm, std_q_norm = q_norm_per_head.mean().item(), q_norm_per_head.std().item()
        mean_k_norm, std_k_norm = k_norm_per_head.mean().item(), k_norm_per_head.std().item()
        mean_v_norm, std_v_norm = v_norm_per_head.mean().item(), v_norm_per_head.std().item()

        print(f"Mean q-norm for steering vector: {mean_q_norm:.3f}, std: {std_q_norm:.3f}")
        print(f"Mean k-norm for steering vector: {mean_k_norm:.3f}, std: {std_k_norm:.3f}")
        print(f"Mean v-norm for steering vector: {mean_v_norm:.3f}, std: {std_v_norm:.3f}")
        
        for head in q_norm_per_head.argsort(descending=True)[:module.num_key_value_groups]:
            print(f"Head {head}: {q_norm_per_head[head]:.3f} (q-norm)")
        for head in k_norm_per_head.argsort(descending=True)[:num_kv_heads // 2]:
            head_idx_start = head * module.num_key_value_groups
            head_idx_end = head_idx_start + module.num_key_value_groups
            print(f"Head ({head_idx_start}-{head_idx_end}): {k_norm_per_head[head]:.3f} (k-norm)")
        for head in v_norm_per_head.argsort(descending=True)[:num_kv_heads // 2]:
            head_idx_start = head * module.num_key_value_groups
            head_idx_end = head_idx_start + module.num_key_value_groups
            print(f"Head ({head_idx_start}-{head_idx_end}): {v_norm_per_head[head]:.3f} (v-norm)")

        plot_attention_heads(q_norm_per_head, mean_q_norm, std_q_norm, BEST_LAYER, visualize_step, "q-norm")
        plot_attention_heads(k_norm_per_head, mean_k_norm, std_k_norm, BEST_LAYER, visualize_step, "k-norm")
        plot_attention_heads(v_norm_per_head, mean_v_norm, std_v_norm, BEST_LAYER, visualize_step, "v-norm")
        
    return hook

def get_moe_handles(decoder_layer):
    """
    Returns (router_module, gate_linear_or_None, num_experts_or_None).
    - router_module: the Router (e.g., GptOssTopKRouter)
    - gate_linear_or_None: a child nn.Linear that maps H -> E (if found)
    - num_experts_or_None: inferred from experts module or linear.out_features
    """
    router = None
    experts_mod = None
    # 1) find router + experts containers under this layer
    for name, sub in decoder_layer.named_modules():
        lname = name.lower()
        cls = sub.__class__.__name__.lower()
        if router is None and ("router" in lname or "router" in cls):
            router = sub
        if experts_mod is None and ("experts" in lname or "experts" in cls):
            experts_mod = sub

    if router is None:
        return None, None, None

    # 2) find the inner linear that outputs expert scores (H -> E)
    gate_linear, E_best = None, -1
    for n, m in router.named_modules():
        if isinstance(m, nn.Linear):
            # heuristics: pick the largest out_features as candidate E
            if m.out_features > E_best:
                gate_linear, E_best = m, m.out_features

    # 3) infer num_experts
    num_experts = getattr(experts_mod, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(router, "num_experts", None)
    if num_experts is None and gate_linear is not None:
        num_experts = gate_linear.out_features
    if num_experts is None and hasattr(router, "weight"):  # rare
        num_experts = router.weight.shape[0]

    return router, gate_linear, num_experts

def create_moe_capture_hook(storage, layer_idx, tag, step_ref=lambda: None, top_k=2, save_full=False, num_experts=None):
    def hook(module, inputs, output):
        # try to extract logits from various router outputs
        logits = None
        if torch.is_tensor(output):
            logits = output
        elif isinstance(output, (list, tuple)):
            # pick the tensor whose last dim matches num_experts
            for t in output:
                if torch.is_tensor(t) and (num_experts is None or t.shape[-1] == num_experts):
                    logits = t; break
        elif isinstance(output, dict):
            for key in ("logits","scores","router_logits"):
                if key in output and torch.is_tensor(output[key]):
                    logits = output[key]; break
        if logits is None:
            return output  # nothing to record

        if logits.dim() == 3: last = logits[:, -1, :]
        else:                 last = logits[-1:, :]
        probs = F.softmax(last, dim=-1)
        k = min(top_k, last.size(-1))
        topk_logit, topk_idx = torch.topk(last, k=k, dim=-1)
        topk_prob = probs.gather(-1, topk_idx)
        storage.append({
            "tag": tag, "layer": layer_idx, "step": step_ref(),
            "topk_idx": topk_idx.detach().cpu().tolist(),
            "topk_logit": topk_logit.detach().cpu().tolist(),
            "topk_prob": topk_prob.detach().cpu().tolist(),
        })
        return output
    return hook


