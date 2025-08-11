import os
import json
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
    
    context = "Trong năm 2020, công ty XYZ đạt doanh thu 500 triệu USD, tăng lên 1000 triệu USD vào năm 2023. Số lượng sản phẩm bán ra cũng tăng từ 10.000 sản phẩm trong năm 2020 lên 18.000 sản phẩm trong năm 2023."
    
    system_prompt = "Bạn là trợ lý AI hữu ích, chuyên trích xuất các thực thể số từ văn bản và trả về dưới dạng JSON theo schema được cung cấp."

    EXTRACT_PROMPT_TEMPLATE_NUMERICAL = (
        "### Role:\n"
        "Bạn là một chuyên gia phân tích dữ liệu có nhiệm vụ trích xuất số liệu từ văn bản để xây dựng một cơ sở dữ liệu đồ thị (knowledge graph).\n"
        "\n"
        "### Instruction:\n"
        "- Dựa vào văn bản sau đây, hãy trích xuất tất cả **Nodes** (thực thể) và **Relationships** (mối quan hệ) giữa chúng.\n"
        "\n"
        "1.  **Thực thể (Node):**\n"
        '    - `id`: Tên duy nhất của thực thể, chuẩn hóa và đầy đủ.\n'
        '    - `entity_category`: Loại thực thể (ví dụ: "Ngân sách", "Doanh thu", "Tổ chức").\n'
        '    - `unit`: Đơn vị đo lường của số liệu (ví dụ: "tỷ đồng", "triệu USD", "sản phẩm").\n'
        '    - `value`: Giá trị số liệu, dạng số (không kèm đơn vị).\n'
        '    - `year`: Năm của số liệu nếu được đề cập, nếu không thì để null.\n'
        "2.  **Mối quan hệ (Relationship):**\n"
        '    - `source` và `target`: PHẢI khớp chính xác với trường `id` của các node đã được trích xuất.\n'
        '    - `relationship_category`: Loại mối quan hệ giữa 2 thực thể (dùng cụm động từ, ví dụ: \"CÔNG_BỐ_BỞI\", \"GHI_NHẬN_TRONG\").\n'
        "3.  **Tổng quát:**\n"
        '    - Chỉ trích xuất thông tin có trong văn bản. KHÔNG suy diễn hoặc bịa đặt.\n'
        '    - Chỉ trích xuất thông tin có số liệu, Node nào không có số liệu thì không trích xuất.\n'
        '    - Kết quả đầu ra PHẢI là một đối tượng JSON hợp lệ, hãy nhớ rằng đầu ra trả về nằm trong thẻ `<output>`.\n'
        "\n"
        "### Ví dụ định dạng đầu ra:\n"
        "<output>\n"
        '{{"nodes": [{{"id": "entity_name", "entity_category": "entity_category", "unit": "entity_unit", "value": entity_value, "year": entity_year}}],\n'
        '"relationships": [{{"source": "entity_1", "target": "entity_2", "relationship_category": "relationship_category"}}]}}\n'
        "</output>\n"
        "\n"
        "### Thực hiện với input sau:\n"
        "<input>\n"
        "{context}\n"
        "</input>\n"
    )    

    json_schema = {
        "name": "numerical_entities_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Tên duy nhất của thực thể, chuẩn hóa và đầy đủ"
                            },
                            "entity_category": {
                                "type": "string",
                                "description": "Loại của số liệu (ví dụ: 'Ngân sách', 'Doanh thu')"
                            },
                            "unit": {
                                "type": "string",
                                "description": "Đơn vị của số liệu (ví dụ: 'Tỷ đồng', 'Sản phẩm')"
                            },
                            "value": {
                                "type": "number",
                                "description": "Giá trị số liệu (dạng số, không bao gồm đơn vị)"
                            },
                            "year": {
                                "type": ["integer", "null"],
                                "description": "Năm của số liệu nếu được đề cập, nếu không thì để null"
                            },
                        },
                        "required": ["id", "entity_category", "unit", "value", "year"],
                        "additionalProperties": False
                    }
                },
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "id của thực thể nguồn, phải khớp với một node đã trích xuất"
                            },
                            "target": {
                                "type": "string",
                                "description": "id của thực thể đích, phải khớp với một node đã trích xuất"
                            },
                            "relationship_category": {
                                "type": "string",
                                "description": "Loại mối quan hệ giữa hai thực thể (ví dụ: 'CÔNG_BỐ_BỞI', 'GHI_NHẬN_TRONG')"
                            }
                        },
                        "required": ["source", "target", "relationship_category"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["nodes", "relationships"],
            "additionalProperties": False
        },
        "strict": True,
    }
    
    prompt_context = EXTRACT_PROMPT_TEMPLATE_NUMERICAL.format(context=context)
    response = await client.chat.completions.create(
        seed=42,
        temperature=0,
        top_p=0.8,
        max_tokens=4096,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop=["<|im_end|>"],
        model="Qwen/Qwen3-4B-AWQ",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_context}
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