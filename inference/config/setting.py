from typing import List
from pydantic import Field
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class LLMConfig(BaseSettings):
    base_url: str = Field(
        alias="LLM_URL",
        description="Base URL for OpenAI API",
    )
    api_key: str = Field(
        alias="LLM_KEY",
        description="API key for OpenAI",
    )
    model: str = Field(
        default="Qwen/Qwen3-4B-AWQ",
        alias="LLM_MODEL",
        description="Model name to be used with AWQ quantization",
    )
    temperature: float = Field(
        default=0.7,
        alias="TEMPERATURE",
        description="Sampling temperature; higher values make output more random",
    )
    top_p: float = Field(
        default=0.8,
        alias="TOP_P",
        description="Nucleus sampling parameter; higher values increase randomness",
    )
    top_k: int = Field(
        default=20,
        alias="TOP_K",
        description="Top-k sampling parameter; higher values increase randomness",
    )
    gpu_memory_utilization: float = Field(
        default=0.80,
        alias="GPU_MEMORY_UTILIZATION",
        description="GPU memory utilization ratio",
    )
    max_tokens: int = Field(
        default=4096,
        alias="MAX_TOKENS",
        description="Maximum number of tokens for API responses",
    )
    max_model_len: int = Field(
        default=16384,
        alias="MAX_MODEL_LENGTH",
        description="Maximum length of model input tokens and output tokens, can be larger if use YaRN",
        examples=[4096, 8192, 16384, 32768, 131072]
    )
    max_num_batched_tokens: int = Field(
        default=8192,
        alias="MAX_NUM_BATCHED_TOKENS",
        description="Maximum number of tokens to process in a single batch",
    )
    max_num_seqs: int = Field(
        default=4,
        alias="MAX_NUM_SEQS",
        description="Maximum number of sequences to process in parallel",
    )
    stop_tokens: List[str] = Field(
        default=["</s>", "EOS", "<|im_end|>"],
        alias="STOP_TOKENS",
        description="Tokens that indicate the end of a sequence",
    )
    quantization: str = Field(
        default="awq_marlin",
        alias="QUANTIZATION",
        description="Quantization method to be used for the model. e.g. if model is AWQ, must use 'awq' or 'awq_marlin'.",
        examples=["awq", "awq_marlin", "gguf", "bitsandbytes"]
    )
    dtype: str = Field(
        default="float16",
        alias="DATA_TYPE",
        description="Data type for model weights and computations",
    )
    kv_cache_dtype: str = Field(
        default="auto",
        alias="KV_CACHE_DTYPE",
        description="Data type for key-value cache; 'auto' uses the same dtype as the model",
    )
    presence_penalty: float = Field(
        default=0.5,
        alias="PRESENCE_PENALTY",
        description="Penalty for new tokens based on existing ones; higher values discourage repetition",
    )
    frequency_penalty: float = Field(
        default=0.5,
        alias="FREQUENCY_PENALTY",
        description="Penalty for new tokens based on their frequency; higher values discourage frequent tokens",
    )
    task: str = Field(
        default="generate",
        alias="TASK",
        description="Task type for the model; currently only 'generate' is supported",
        examples=["generate", "classify", "embed"]
    )
    seed: int = Field(
        default=42,
        alias="SEED",
        description="Random seed for sampling"
    )


llm_config = LLMConfig()
