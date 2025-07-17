import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1/",
        api_key="your_api_key_here"
    )
    system_prompt = "Bạn là trợ lý AI hữu ích."
    prompt = "Giải thích khái niệm học máy."
    response = await client.chat.completions.create(
        seed=42,
        temperature=0.7,
        top_p=0.95,
        model="Qwen/Qwen3-8B-AWQ",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    print(content)

if __name__ == "__main__":
    asyncio.run(main())