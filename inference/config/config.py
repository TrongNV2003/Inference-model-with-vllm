from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union

class InferenceRequest(BaseModel):
    prompt: str

class InferenceResponse(BaseModel):
    text: str
    finish_reason: str
    tokens_used: int

class ChatMessage(BaseModel):
    role: str
    content: str

class ResponseFormat(BaseModel):
    type: str
    json_schema: Optional[Union[str, Dict[str, Any]]] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    seed: int = 42
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: int = 0
    max_tokens: int = 4096
    min_tokens: Optional[int] = 0
    presence_penalty: float = 0.5
    frequency_penalty: float = 0.5
    stop_tokens: List[str] = ["</s>", "EOS", "<|im_end|>"]
    response_format: Optional[Union[Dict[str, Any], ResponseFormat]] = None
    stream: Optional[bool] = True

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
