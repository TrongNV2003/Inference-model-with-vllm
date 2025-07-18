import os
import requests

api_url = os.getenv("LLM_URL")
api_key = os.getenv("LLM_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "prompt": "Xin chào, hãy giải thích về AI là gì?"
}

response = requests.post(api_url, headers=headers, json=data)

if response.status_code == 200:
    results = response.json()
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Finish reason: {result['finish_reason']}")
        print(f"Tokens used: {result['tokens_used']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
    
    
"""
curl -X POST http://localhost:8000/v1/ \
  -H "Authorization: Bearer your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Xin chào, hãy giải thích về AI là gì?"
  }'
"""