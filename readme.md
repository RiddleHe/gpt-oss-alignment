# GPT-OSS-Alignment

This repo is under active development to do a bunch of interp / alignment stuff on gpt-oss-20b

## Set up

### Conda Env
```
conda create -n gptoss python=3.12 -y
conda activate gptoss

conda install -c conda-forge libstdcxx-ng -y

pip install uv
    
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match
uv pip install openai --upgrade
```

### Sample from vLLM

```
bash vllm.sh
# then play with sample_gpt.ipynb
```

## Steering Vector

This pipeline computes the mean difference between activation vectors of unaligned responses and those of aligned responses. It then uses these activation vector to steer the outputs of the model. (Cool observation: I compute the activation of the prompt, not response!)

### Code

I will make these scripts more production grade in the coming days.

```
export CUDA_VISIBLE_DEVICES=[your device] # most stable on single device
python find_vector.py
python compute_vector_diff.py
python steer_model.py \
    --method "layer|logit" \
    --layer <attack_layer> \
    --visualize_step <step_to_visualize_attention_scores> 
python analyze_vectors.py 
```

Enjoy!