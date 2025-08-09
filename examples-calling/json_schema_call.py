import os
import json
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LLM_KEY")

async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:5000/v1/",
        api_key=api_key
    )
    
    system_prompt = "Bạn là trợ lý AI hữu ích, chuyên trích xuất các thực thể số từ văn bản và trả về dưới dạng JSON theo schema được cung cấp."
    context = (
        "Trong năm 2020, công ty XYZ đạt doanh thu 500 triệu USD, tăng lên 1000 triệu USD vào năm 2023. "
        "Số lượng sản phẩm bán ra cũng tăng từ 10.000 sản phẩm trong năm 2020 lên 18.000 sản phẩm trong năm 2023."
    )
    prompt = (
        f"Trích xuất các thực thể số từ đoạn văn bản sau và trả về dưới dạng JSON theo schema được cung cấp:\n\n"
        f"{context}"
    )

    json_schema = {
        "name": "numerical_entities_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Tên của thực thể số (ví dụ: Doanh thu, Số lượng sản phẩm bán ra)"},
                            "values": {"type": "array", "items": {"type": "number"}, "description": "Danh sách các giá trị số được trích xuất"},
                            "year": {"type": "array", "items": {"type": "integer"}, "description": "Danh sách các năm tương ứng với các giá trị"},
                            "unit": {"type": "string", "description": "Đơn vị của giá trị (ví dụ: Triệu USD, Sản phẩm)"},
                            "description": {"type": "string", "description": "Mô tả ngắn về thông tin của thực thể số đó (ví dụ: Doanh thu của công ty XYZ)"}
                        },
                        "required": ["name", "values", "year", "unit", "description"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["data"],
            "additionalProperties": False
        }
    }

    response = await client.chat.completions.create(
        seed=42,
        temperature=0.7,
        top_p=0.8,
        max_tokens=2048,
        frequency_penalty=0.5,
        presence_penalty=1.5,
        stop=["<|im_end|>"],
        model="Qwen/Qwen3-4B-AWQ",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        stream=False,
        response_format={
            "type": "json_schema",
            "json_schema": json_schema
        },
    )
    content = response.choices[0].message.content
    print(content)

if __name__ == "__main__":
    asyncio.run(main())