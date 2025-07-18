from typing import List, Dict
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
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 1024
    stop: List[str] = ["</s>", "EOS", "<|im_end|>"]
    seed: int = 42
    response_format: Dict[str, str] = None

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