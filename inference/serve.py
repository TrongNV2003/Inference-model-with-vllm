import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

import re
import time
import json
import logging
import uvicorn
import numpy as np
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

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
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
    },
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
    template_kwargs = request.chat_template_kwargs or {}
    
    if credentials.credentials != llm_config.api_key:
        logger.warning("Invalid API key attempt")
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        messages_as_dicts = [msg.model_dump() for msg in request.messages]
        prompt = tokenizer.apply_chat_template(
            messages_as_dicts,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs
        )
        logger.info(f"Formatted Prompt: {prompt}")
        
        if request.response_format and request.response_format.get("type") == "json_object":
            json_instruction = (
                "\n\nIMPORTANT: You must provide a response in a valid JSON format. "
                "Do not include any other text, explanations, or markdown formatting outside of the JSON object. "
                "The JSON object must start with { and end with }."
            )
            prompt += json_instruction
        logger.info(f"Final Formatted Prompt: {prompt[:300]}...")

        # Check if the total number of input + output tokens exceeds the max_model_len limit
        max_tokens = request.max_tokens
        prompt_tokens = len(tokenizer.encode(prompt))
        if prompt_tokens + max_tokens > llm_config.max_model_len:
            raise HTTPException(status_code=400, detail="Tokens exceeds max_model_len")
        
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)  # Nếu chỉ dùng GPU 0
        info = nvmlDeviceGetMemoryInfo(handle)
        total = info.total / 1024**3
        used = info.used / 1024**3
        free = info.free / 1024**3
        logger.info(f"GPU Memory - Total: {total:.2f} GB, Used: {used:.2f} GB, Free: {free:.2f} GB")
        available_vram = free
        if available_vram < 0.5:
            logger.warning("Low VRAM, reducing max_tokens to 512")
            max_tokens = min(max_tokens, 512)
        
        sampling_params = SamplingParams(
            seed=request.seed,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=max_tokens,
            stop=request.stop_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
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

        if request.response_format and request.response_format.get("type") == "json_object":
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if not json_match:
                    raise json.JSONDecodeError("No JSON object found in response", response_text, 0)
                
                json_string = json_match.group(0)
                response_text = json.dumps(json.loads(json_string), ensure_ascii=False)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse model output as JSON. Output: '{response_text}'. Error: {e}")
                raise HTTPException(status_code=500, detail="Model output is not valid JSON.")
        
        completion_tokens = len(tokenizer.encode(response_text))

        # Response in OpenAI API format
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