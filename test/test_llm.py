import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

chat = client.chat.completions.create(
    model=os.getenv("OPENROUTER_MODEL_NAME", "openai/gpt-oss-120b:free"),
    messages=[
        {"role": "system", "content": "Ты дружелюбный ассистент. Отвечай кратко."},
        {"role": "user", "content": "Привет! Расскажи, что ты умеешь."}
    ],
    temperature=0.7,
    max_tokens=300,
    timeout=15,
)

print(chat.choices[0].message.content)