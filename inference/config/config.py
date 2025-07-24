from typing import List, Dict, Optional
from pydantic import BaseModel

class InferenceRequest(BaseModel):
    prompt: str

class InferenceResponse(BaseModel):
    text: str
    finish_reason: str
    tokens_used: int

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    seed: int = 42
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 2048
    presence_penalty: float = 0.5
    frequency_penalty: float = 0.5
    stop_tokens: List[str] = ["</s>", "EOS", "<|im_end|>"]
    response_format: Dict[str, str] = None
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