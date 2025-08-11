import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LLM_KEY")
base_url = os.getenv("LLM_URL")

async def main():
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )
    system_prompt = "Bạn là trợ lý AI hữu ích."
    prompt = "Giải thích khái niệm học máy một cách chi tiết."

    stream_response = await client.chat.completions.create(
        seed=42,
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048,
        frequency_penalty=0.5,
        presence_penalty=1.5,
        stop=["</s>", "EOS", "<|im_end|>"],
        model="Qwen/Qwen3-4B-AWQ",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        stream=True,
    )

    print("AI: ", end="")
    async for chunk in stream_response:
        content_delta = chunk.choices[0].delta.content
        if content_delta is not None:
            print(content_delta, end="", flush=True)

    print()

if __name__ == "__main__":
    asyncio.run(main())