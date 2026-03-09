import os
from dotenv import load_dotenv
import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
import json
import re

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

# Создаём класс для CLI-бота
class CliBot:
    def __init__(self, model_name, system_prompt="Ты полезный ассистент"):
        # Создание модели
        self.chat_model = ChatOpenAI(
            model=model_name,
            api_key=API_KEY,
            temperature=0,
            timeout=15,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "Authorization": f"Bearer {API_KEY}",
                "HTTP-Referer": "http://localhost",
            },
            max_tokens=600
        )

        # Создание хранилища истории
        self.store = {}

        # Создание шаблона промпта
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt), # Добавим возможность менять системный промпт
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        # Создание цепочки
        self.chain = self.prompt | self.chat_model

        # Создание цепочки с историей
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,  # Цепочка с историей
            self.get_session_history,  # Метод для получения истории
            input_messages_key="question",  # Ключ для вопроса
            history_messages_key="history",  # Ключ для истории
        )

        # Загрузка JSON-файлов
        self.faq_data = JsonReader.load_json("data/faq.json", default=[])
        self.orders_data = JsonReader.load_json("data/orders.json", default={})

        # Логирование
        logging.basicConfig(
            filename="chat_session.log", level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
            encoding="utf-8"
        )

    # Метод для получения истории по session_id
    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]


    def find_in_faq(self, user_question: str, faq_questions):
        user_question = user_question.lower().strip()
        user_question = re.sub(r"[^\w\s]", "", user_question)
        user_question_clean = str(user_question)
        for i in range(len(faq_questions)):
            if user_question_clean in faq_questions[i]["q"].lower():
                return faq_questions[i]["a"]

    def order_answer(self, user_question: str, orders):
        if user_question.startswith("/order"):
            user_question = user_question.split()

            if len(user_question) < 2:
                return f"Укажите номер заказа. Пример: /order 55555"

            order_id = user_question[1]
            if order_id in orders:
                status = orders[order_id].get("status", "неизвестен")
                return f"Заказ #{order_id}: {status}"
            else:
                return f"Заказ №{order_id} не найден. Проверьте номер."

    def __call__(self, session_id):
        print(
            "Чат-бот запущен! Можете задавать вопросы. \n "
            "- Для выхода введите 'выход'.\n"
            "- Для очистки контекста введите 'сброс'.\n"
        )
        logging.info("=== New session ===")
        while True:
            try:
                user_text = input("Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                break
            if not user_text:
                continue

            logging.info(f"User: {user_text}")
            msg = user_text.lower()
            if msg in ("выход", "стоп", "конец"):
                print("Бот: До свидания!")
                logging.info("Пользователь завершил сессию. Сессия окончена.")
                break
            if msg == "сброс":
                if session_id in self.store:
                    del self.store[session_id]
                print("Бот: Контекст диалога очищен.")
                logging.info("Пользователь сбросил контекст.")
                continue

            order_answer = self.order_answer(user_text, self.orders_data)

            if order_answer:
                print(f"Бот: {order_answer}")
                logging.info(f"Bot [Order]: {order_answer}")
                continue

            faq_answer = self.find_in_faq(user_text, self.faq_data)

            if faq_answer:
                print(f"Бот: {faq_answer}")
                logging.info(f"Bot [FAQ]: {faq_answer}")
                continue

            try:
                responce = self.chain_with_history.invoke(
                    {"question": user_text},
                    {"configurable": {"session_id": session_id}}
                )
            except Exception as e:
                # Логируем и выводим ошибку, продолжаем чат
                logging.error(f"[error] {e}")
                print(f"[Ошибка] {e}")
                continue

            # Форматируем и выводим ответ
            bot_reply = responce.content.strip()
            logging.info(f"Bot: {bot_reply}")
            print(f"Бот: {bot_reply}")

class JsonReader:
    def __init__(self, file_path):
        self.file_path = file_path

    @staticmethod
    def load_json(file_path, default=None):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f" Файл {file_path} не найден")
            return default

if __name__ == "__main__":
    model = os.getenv("OPENROUTER_MODEL_NAME")
    system_propmpt = """Ты полезный ассистент. Ты всегда дружедюбен и вежлив. Отвечай подробно и по существу"""

    bot = CliBot(
        model_name=model,
        system_prompt=system_propmpt
    )
    bot("user111")