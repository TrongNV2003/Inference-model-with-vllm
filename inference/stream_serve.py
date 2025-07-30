import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

import re
import time
import json
import torch
import logging
import uvicorn
import numpy as np
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from fastapi.responses import StreamingResponse
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

app = FastAPI(title="vLLM Streaming Inference API")

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


async def stream_generator(prompt: str, sampling_params: SamplingParams, request: ChatCompletionRequest, prompt_tokens: int):
    """Tạo streaming response cho chat completion"""
    request_id = f"chatcmpl-{str(np.random.randint(1e9))}"
    created_time = int(time.time())
    
    previous_text = ""
    
    first_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None
            }
        ]
    }
    yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

    final_output = None
    async for output in engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id
    ):
        final_output = output
        current_text = output.outputs[0].text
        delta_text = current_text[len(previous_text):]
        previous_text = current_text

        if delta_text:
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": delta_text},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    
    if final_output:
        completion_tokens = len(tokenizer.encode(final_output.outputs[0].text))
        finish_reason = final_output.outputs[0].finish_reason
        
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
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
            enable_thinking=request.enable_thinking,
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
        max_tokens = llm_config.max_tokens
        prompt_tokens = len(tokenizer.encode(prompt))
        if prompt_tokens + max_tokens > llm_config.max_model_len:
            raise HTTPException(status_code=400, detail="Tokens exceeds max_model_len")
        
        available_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
        if available_vram < 1:
            logger.warning("Low VRAM, reducing max_tokens")
            max_tokens = min(max_tokens, 512)
        
        sampling_params = SamplingParams(
            seed=request.seed,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            stop=request.stop_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
        )
        if request.stream:
            return StreamingResponse(
                stream_generator(prompt, sampling_params, request, prompt_tokens), 
                media_type="text/event-stream"
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