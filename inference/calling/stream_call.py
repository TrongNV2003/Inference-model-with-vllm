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
    prompt = "Giải thích khái niệm học máy một cách chi tiết."

    stream_response = await client.chat.completions.create(
        seed=42,
        temperature=0.7,
        top_p=0.95,
        model="Qwen/Qwen3-4B",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        stream=True,
        # response_format={"type": "json_object"}
    )

    print("AI: ", end="")
    async for chunk in stream_response:
        content_delta = chunk.choices[0].delta.content
        if content_delta is not None:
            print(content_delta, end="", flush=True)

    print()

if __name__ == "__main__":
    asyncio.run(main())