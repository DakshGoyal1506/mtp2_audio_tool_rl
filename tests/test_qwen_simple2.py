import requests
import json

url = "http://192.168.1.30:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "Qwen/Qwen3.5-27B",
    "messages": [
        {"role": "user", "content": "Explain the importance of tempo in music in exactly three sentences."}
    ],
    "temperature": 0.2,
    "max_tokens": 150
}

response = requests.post(url, headers=headers, json=data)
print(json.dumps(response.json(), indent=2))
