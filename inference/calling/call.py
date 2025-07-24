import os
import asyncio
from openai import AsyncOpenAI

api_key = os.getenv("LLM_KEY")

async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1/",
        api_key=api_key
    )
    system_prompt = "Bạn là trợ lý AI hữu ích."
    prompt = "Giải thích khái niệm học máy."
    response = await client.chat.completions.create(
        seed=42,
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop=["<|im_end|>"],
        model="Qwen/Qwen3-4B-AWQ",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        stream=False
    )
    content = response.choices[0].message.content
    print(content)

if __name__ == "__main__":
    asyncio.run(main())