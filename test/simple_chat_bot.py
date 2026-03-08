import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL_NAME")

model = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "http://localhost",
    },
    max_tokens=300
)

agent = create_agent(
    model=model,
    checkpointer=InMemorySaver(),
)

conf = {"configurable": {"thread_id": "conversation_001"}}

print(f"Введите ваш запрос боту и дождитесь ответа (введите 'выход' для выхода): \n")

while True:
    question = input("Вы: ").strip()

    if not question:
        continue

    if question.lower() in ["exit", "выход", "q", "quit"]:
        print("Бот: До свидания!")
        break

    try:
        responce = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=conf
        )

        answer = responce["messages"][-1].content
        print(f"Бот: {answer}\n")

    except Exception as e:
        print(f"Ошибка: {e}")