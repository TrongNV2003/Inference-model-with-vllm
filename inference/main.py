import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

import time
import json
import torch
import logging
import uvicorn
import numpy as np
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from inference.config.setting import llm_config
from inference.config.config import (
    ChatMessage,
    ChatCompletionUsage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

security = HTTPBearer()
app = FastAPI(title="vLLM Inference API")

engine_args = AsyncEngineArgs(
    seed=llm_config.seed,
    model=llm_config.model,
    quantization="awq_marlin",
    gpu_memory_utilization=llm_config.gpu_memory_utilization,
    tensor_parallel_size=1,
    max_model_len=llm_config.max_model_length,
    max_num_seqs=llm_config.max_num_seqs,
    max_num_batched_tokens=llm_config.max_num_batched_tokens,
    enforce_eager=False,
    enable_prefix_caching=True,
    dtype="float16",
    device="cuda",
    kv_cache_dtype="auto",
    task="generate",
)

try:
    logger.info("Loading engine...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(llm_config.model, use_fast=True, trust_remote_code=True)
    
    logger.info("vLLM Engine and tokenizer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize engine or tokenizer: {str(e)}")
    raise


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(
    request: ChatCompletionRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """OpenAI-compatible chat completion API"""
    if credentials.credentials != llm_config.api_key:
        logger.warning("Invalid API key attempt")
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        messages_as_dicts = [msg.model_dump() for msg in request.messages]
        prompt = tokenizer.apply_chat_template(
            messages_as_dicts,
            tokenize=False,
            add_generation_prompt=True,
        )
        logger.info(f"Formatted Prompt: {prompt}")
        
        prompt_tokens = len(tokenizer.encode(prompt))
        max_tokens = llm_config.max_tokens
        
        if prompt_tokens + max_tokens > llm_config.max_model_length:    # Kiểm tra xem tổng số tokens có vượt quá giới hạn không
            raise HTTPException(status_code=400, detail="Prompt + max_tokens exceeds max_model_len")
        
        available_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
        if available_vram < 1:
            logger.warning("Low VRAM, reducing max_tokens")
            max_tokens = min(max_tokens, 512)
        
        sampling_params = SamplingParams(
            seed=llm_config.seed,
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            top_k=llm_config.top_k,
            max_tokens=max_tokens,
            stop=llm_config.stop_tokens,
            presence_penalty=0.5,
            frequency_penalty=0.5
        )

        final_output = None
        async for output in engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=str(np.random.randint(1e9))
        ):
            final_output = output
            
        if final_output is None:
            raise HTTPException(status_code=500, detail="Failed to generate response")

        response_text = final_output.outputs[0].text.strip()
        completion_tokens = len(tokenizer.encode(response_text))

        if request.response_format and request.response_format.get("type") == "json_object":
            try:
                response_text = json.dumps(json.loads(response_text), ensure_ascii=False)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Model output is not valid JSON")
            
        # Tạo phản hồi theo định dạng OpenAI API
        response = ChatCompletionResponse(
            id=f"chatcmpl-{str(np.random.randint(1e9))}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_text
                    ),
                    finish_reason=final_output.outputs[0].finish_reason
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
        return response

    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

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
