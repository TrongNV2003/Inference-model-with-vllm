import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

import time
import json
import logging
import numpy as np
from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from jsonschema import validate, ValidationError
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Security, Request
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

router = APIRouter(tags=["vLLM Serving"])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
security = HTTPBearer()


nvmlInit()
nvml_handle = nvmlDeviceGetHandleByIndex(0)  # GPU 0


DEFAULT_JSON_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": True
}

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.post("/v1/chat/completions")
async def chat_completion(
    fastapi_request: Request,
    request: ChatCompletionRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """OpenAI-compatible chat completion API"""
    engine = fastapi_request.app.state.engine
    tokenizer = fastapi_request.app.state.tokenizer
    
    if credentials.credentials != llm_config.api_key:
        logger.warning("Invalid API key attempt")
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        messages_as_dicts = [msg.model_dump() for msg in request.messages]
        prompt = tokenizer.apply_chat_template(
            messages_as_dicts,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        logger.info(f"Formatted Prompt: {prompt}")
        
        guided_decoding_params = None
        schema = None
        
        if request.response_format and request.response_format.get("type") == "json_schema":
            json_schema_data = request.response_format.get("json_schema")
            if json_schema_data:
                if isinstance(json_schema_data, dict):
                    schema = json_schema_data.get("schema")
                elif isinstance(json_schema_data, str):
                    try:
                        parsed_schema = json.loads(json_schema_data)
                        schema = parsed_schema.get("schema")
                    except json.JSONDecodeError:
                        schema = None
            if not schema:
                raise HTTPException(status_code=400, detail="JSON schema is required for json_schema response format")
            try:
                guided_decoding_params = GuidedDecodingParams(json=schema)
                logger.info("Applied JSON schema enforcement with guided decoding")
            except Exception as e:
                logger.error(f"Failed to initialize guided decoding with schema: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid JSON schema: {str(e)}")
        
        elif request.response_format and request.response_format.get("type") == "json_object":
            try:
                guided_decoding_params = GuidedDecodingParams(json_object=True)
                logger.info("Applied JSON object enforcement with guided decoding")
            except Exception as e:
                logger.error(f"Failed to initialize guided decoding for JSON object: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid JSON configuration: {str(e)}")


        # Check if the total number of input + output tokens exceeds the max_model_len limit
        max_tokens = request.max_tokens
        prompt_tokens = len(tokenizer.encode(prompt))
        if prompt_tokens + max_tokens > llm_config.max_model_len:
            raise HTTPException(status_code=400, detail="Tokens exceeds max_model_len")
        
        info = nvmlDeviceGetMemoryInfo(nvml_handle)
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
            min_tokens=request.min_tokens,
            stop=request.stop_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            guided_decoding=guided_decoding_params,
        )

        async def model_generator(request_id: str):
            async for output in engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id
            ):
                yield output

        if request.stream:
            async def stream_generator():
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
                async for output in model_generator(request_id):
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
            
            return StreamingResponse(
                stream_generator(), media_type="text/event-stream"
            )
            
        else:
            final_output = None
            request_id = f"chatcmpl-{str(np.random.randint(1e9))}"
            async for output in model_generator(request_id):
                final_output = output
            
            if final_output is None:
                raise HTTPException(status_code=500, detail="Failed to generate response")

            response_text = final_output.outputs[0].text.strip()

            if request.response_format and request.response_format.get("type") in ["json_object", "json_schema"]:
                try:
                    json_response = json.loads(response_text)
                    if schema and request.response_format.get("type") == "json_schema":
                        try:
                            validate(instance=json_response, schema=schema)
                            logger.info("Model output validated against JSON schema")
                        except ValidationError as e:
                            logger.error(f"JSON does not match schema: {str(e)}")
                            raise HTTPException(status_code=500, detail=f"JSON does not match schema: {str(e)}")
                    response_text = json.dumps(json_response, ensure_ascii=False)
                    
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
