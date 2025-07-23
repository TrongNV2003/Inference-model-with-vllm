# Inference Large language models with vLLM
This repo implements vLLM to self-hosted LLM on your own devices.
Model usage: Qwen/Qwen3-8B-AWQ
Inference device: GPU Nvidia RTX 5060ti 16GB

## Installation dependancies
```
conda create -n inference python==3.11.11
conda activate inference
```

```sh
pip install -r requirements.txt
```

## Execution
Serving LLM with vLLM:
```python
python inference/main.py
```

Call API:
```python
python inference/calling/call.py
```

### Note:
- Check your TORCH_CUDA_ARCH_LIST of GPU at [Here](https://developer.nvidia.com/cuda-gpus#compute)
- Check config settings RoPE for long Texts at [YaRN](https://huggingface.co/Qwen/Qwen3-4B-AWQ#:~:text=Qwen3%20natively%20supports,the%20YaRN%20method.)
- Check max context length (max_model_len) of models (Qwen3-4B, Qwen3-8B...) at [Here](https://qwenlm.github.io/blog/qwen3/#advanced-usages:~:text=We%20are%20open,Apache%202.0%20license.)

## Future plans
- Bổ sung trigger bỏ thinking mode với model Qwen3 (hiện tại default model là True) (DONE)
- Bổ sung output json format
