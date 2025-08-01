# Inference Large language models with vLLM
This repo implements vLLM to self-hosted LLM on your own devices.
Model usage: Qwen/Qwen3-4B-AWQ
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
python inference/app.py
```

Call API:
Calling API after serving:
```python
python inference/calling/call.py
```

Calling API after streaming serving:
```python
python inference/calling/stream_call.py
```

Quick call:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Xin chào, hãy giải thích về AI là gì?"
      }
    ],
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": 8192,
    "presence_penalty": 1.5
  }'
```

### Note:
- Check your TORCH_CUDA_ARCH_LIST of GPU at [Here](https://developer.nvidia.com/cuda-gpus#compute)
- Check config settings RoPE for long Texts at [YaRN](https://huggingface.co/Qwen/Qwen3-4B-AWQ#:~:text=Qwen3%20natively%20supports,the%20YaRN%20method.)
- Check max context length (max_model_len) of models (Qwen3-4B, Qwen3-8B...) at [Here](https://qwenlm.github.io/blog/qwen3/#advanced-usages:~:text=We%20are%20open,Apache%202.0%20license.)
- Reference to hyperparameters set for Thinking-mode and Non Thinking-mode of Qwen models from [Qwen](https://huggingface.co/Qwen/Qwen3-4B-AWQ#best-practices)

### vLLM features:
Apply YaRN for extend max-model-len for LLM:
- "original_max_position_embeddings": 32768. This is the length of the original “ruler”. It tells the algorithm that this Qwen3-4B model was originally trained to understand a maximum of 32,768 tokens.
- "factor": 4.0. This is the "compression factor". It says "Compress the divisions by 4 times".
- YaRN (Yet another RoPE extensioN method): This is a smarter approach. It “compresses” unevenly. At early positions (near context), it compresses less to retain detail. At distant positions, it compresses more. This allows the model to handle long text without losing the ability to understand nearby details.
- The new context window length = original_max_position_embeddings * factor = 32768 * 4.0 = 131072. This number exactly matches the --max-model-len 131072 parameter from Qwen suggestion.


## Future plans
- Bổ sung trigger bỏ thinking mode với model Qwen3 (hiện tại default model là True) (DONE)
- Bổ sung output json format (DONE)
