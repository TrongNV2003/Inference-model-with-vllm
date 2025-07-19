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
- Check your TORCH_CUDA_ARCH_LIST at [Here](https://developer.nvidia.com/cuda-gpus#compute)

## Future plans
- TBD
