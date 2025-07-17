import torch
import asyncio
import logging
import uvicorn
import numpy as np
from typing import List
from transformers import AutoTokenizer
from contextlib import asynccontextmanager
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from inference.config.setting import llm_config
from inference.config.config import (
    InferenceRequest,
    InferenceResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

security = HTTPBearer()

app = FastAPI(title="vLLM Inference API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý lifecycle của engine"""
    try:
        logger.info("Initializing vLLM Async Engine...")
        engine_args = AsyncEngineArgs(
            seed=llm_config.seed,
            model=llm_config.model,
            quantization="awq",
            gpu_memory_utilization=llm_config.gpu_memory_utilization,
            tensor_parallel_size=1,
            max_model_len=llm_config.max_model_length,
            max_num_seqs=llm_config.max_num_seqs,
            max_num_batched_tokens=llm_config.max_num_batched_tokens,
            enforce_eager=True,  # Tắt graph compilation để ổn định trên 16GB VRAM
            enable_prefix_caching=True,  # Tăng tốc với prefix caching
            dtype="float16",
            kv_cache_dtype="fp8",
        )
        global engine
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        logger.info("Loading tokenizer...")
        global tokenizer
        model_path = llm_config.model
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        logger.info("Tokenizer loaded successfully")
        
        logger.info("vLLM Engine initialized successfully")
        yield
    finally:
        logger.info("Shutting down vLLM Engine...")
        del tokenizer
        torch.cuda.empty_cache()

app.lifespan = lifespan


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/v1/completions", response_model=List[InferenceResponse])
async def generate_completion(
    request: InferenceRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Xác thực API key"""
    if credentials.credentials != llm_config.api_key:
        logger.warning("Invalid API key attempt")
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        """ Kiểm tra độ dài prompt để tránh lỗi OOM """
        if len(request.prompt) > llm_config.max_model_length:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt too long, maximum allowed: {llm_config.max_model_length} tokens"
            )
            
        sampling_params = SamplingParams(
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            top_k=llm_config.top_k,
            max_tokens=llm_config.max_tokens,
            stop=llm_config.stop_tokens,  # Dừng khi gặp token kết thúc
            presence_penalty=0.5,  # Khuyến khích nội dung mới
            frequency_penalty=0.5  # Giảm lặp từ
        )

        outputs = []
        async for output in engine.generate(
            prompt=request.prompt,
            sampling_params=sampling_params,
            request_id=str(np.random.randint(1e9))
        ):
            outputs.append(output)

        results = [
            InferenceResponse(
                text=output.outputs[0].text,
                finish_reason=output.outputs[0].finish_reason,
                tokens_used=len(output.outputs[0].token_ids)
            )
            for output in outputs
        ]
        return results

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

def run_server():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_keep_alive=30
    )

if __name__ == "__main__":
    import threading
    server_thread = threading.Thread(target=run_server)
    server_thread.start()
