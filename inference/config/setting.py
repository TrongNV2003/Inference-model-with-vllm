from typing import List
from pydantic import Field
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class LLMConfig(BaseSettings):
    base_url: str = Field(
        description="Base URL for OpenAI API",
        alias="LLM_URL",
    )
    api_key: str = Field(
        description="API key for OpenAI",
        alias="LLM_KEY",
    )
    model: str = Field(
        default="Qwen/Qwen3-4B-AWQ",
        description="Model name to be used with AWQ quantization",
        alias="LLM_MODEL",
    )
    max_tokens: int = Field(
        default=2048,
        alias="MAX_TOKENS",
        description="Maximum number of tokens for API responses",
    )
    temperature: float = Field(
        default=0.6,
        description="Sampling temperature; higher values make output more random",
        alias="TEMPERATURE",
    )
    top_p: float = Field(
        default=0.95,
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
    max_model_length: int = Field(
        default=4096,
        alias="MAX_MODEL_LENGTH",
        description="Maximum length of the model input",
    )
    max_num_batched_tokens: int = Field(
        default=4096,
        alias="MAX_NUM_BATCHED_TOKENS",
        description="Maximum number of tokens to process in a single batch",
    )
    max_num_seqs: int = Field(
        default=32,
        alias="MAX_NUM_SEQS",
        description="Maximum number of sequences to process in parallel",
    )
    stop_tokens: List[str] = Field(
        default=["</s>", "EOS", "<|im_end|>"],
        alias="STOP_TOKENS",
        description="Tokens that indicate the end of a sequence",
    )
    seed: int = Field(
        default=42,
        alias="SEED",
        description="Random seed for sampling"
    )


llm_config = LLMConfig()
