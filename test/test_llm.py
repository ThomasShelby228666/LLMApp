import os
from dotenv import load_dotenv
from openai import OpenAI
import sys
import argparse

def main():
    # 1) Загрузка переменных окружения
    load_dotenv()

    # 2) Аргументы поиска
    parser = argparse.ArgumentParser(
        description="Первый запрос к LLM"
    )
    parser.add_argument(
        "-q", "--query",
        default="Привет, назови столицу России.",
        help="Текст запроса к модели (по умолчанию - простой вопрос)"
    )
    args = parser.parse_args()

    # 3) Создание клиента
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    # 4) Минимальный запрос
    chat = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL_NAME", "openai/gpt-oss-120b:free"),
        messages=[
            {"role": "system", "content": "Ты дружелюбный ассистент. Отвечай кратко."},
            {"role": "user", "content": args.query}
        ],
        temperature=0.7,
        max_tokens=300,
        timeout=15,
    )

    # 5) Печатаем ответ
    print(chat.choices[0].message.content)

    return 0

if __name__ == "__main__":
    sys.exit(main())