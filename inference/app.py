import asyncio
from loguru import logger
from fastapi import FastAPI
from vllm import AsyncLLMEngine
from transformers import AutoTokenizer
from contextlib import asynccontextmanager
from vllm.engine.arg_utils import AsyncEngineArgs
from fastapi.middleware.cors import CORSMiddleware

from inference.routes.routes import router
from inference.config.setting import llm_config

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup sequence starting...")

    engine_args = AsyncEngineArgs(
        seed=llm_config.seed,
        model=llm_config.model,
        task=llm_config.task,
        quantization=llm_config.quantization,
        gpu_memory_utilization=llm_config.gpu_memory_utilization,
        tensor_parallel_size=1,
        max_model_len=llm_config.max_model_len,
        max_num_seqs=llm_config.max_num_seqs,
        max_num_batched_tokens=llm_config.max_num_batched_tokens,
        enforce_eager=False,    # Use CUDA graph for performance, may reduce latency
        enable_prefix_caching=True,
        dtype=llm_config.dtype,
        kv_cache_dtype=llm_config.kv_cache_dtype,
        device="auto",
        rope_scaling={
            "type": "yarn",
            "factor": 2.0, # Scale factor for rope (compression factor), e.g. max_model_len = 32768 *4.0 = 131072 tokens
            "original_max_position_embeddings": 32768,
        },
    )

    logger.info("Loading engine...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
        
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(llm_config.model, use_fast=True, trust_remote_code=True)
    app.state.engine = engine
    app.state.tokenizer = tokenizer
    
    logger.info("vLLM Engine and tokenizer initialized successfully")

    yield

    logger.info("Lifespan: Shutdown sequence starting...")
    del app.state.engine
    del app.state.tokenizer
    logger.info("Lifespan: Resources cleaned up.")


app = FastAPI(title="vLLM Inference API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

