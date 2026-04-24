# app.py
import sys
from pathlib import Path
import gradio as gr

# 1. Настройка путей
project_root = Path(__file__).resolve().parent
src_path = project_root / "src"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# 2. Импорт компонентов
from config import QDRANT_COLLECTION
from src.rag_chain import RAGChain

# Инициализация модели
print("Загрузка моделей и подключение к Qdrant")
try:
    rag = RAGChain(collection_name=QDRANT_COLLECTION)
    print("Система готова")
except Exception as e:
    print(f"Ошибка запуска: {e}")
    raise e


def respond(
    message: str,
    history: list | None  # или Optional[list]
) -> tuple[str, list]:
    """
    Обработчик сообщений для чат-интерфейса Gradio.
    """
    if not message:
        return "", history

    # Ответ от RAG
    result = rag.ask_rag(message, top_k=3)

    answer = result["answer"]
    confidence = result["confidence"]
    citations = result["citations"]

    # Красивый ответ с Markdown
    conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}
    conf_text = f"{conf_emoji.get(confidence, '⚪')} Уверенность: **{confidence.upper()}**"

    sources_md = ""
    if citations:
        sources_md = "\n\n---\n### Источники:\n"

        for i, c in enumerate(citations, 1):
            authors = c["authors"]
            has_valid_authors = authors and authors != "Автор неизвестен"

            if has_valid_authors:
                source_name = authors
            else:
                source_name = c["source"].split("/")[-1]

            year = c["year"] or "н.д."
            snippet = c["snippet"].replace("...", "")

            source_line = f"{i}. **{source_name} ({year})**\n> {snippet}\n\n"
            sources_md += source_line
    else:
        sources_md = "\n\n---\n*Источники не найдены в базе знаний.*"

    full_response = f"{answer}\n\n{conf_text}{sources_md}"

    if history is None:
        history = []

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": full_response})

    return "", history


# Создание интерфейса
with gr.Blocks(title="Научный ассистент") as demo:
    gr.Markdown("# Научный Ассистент")
    gr.Markdown("""
    Задавайте вопросы по загруженным научным статьям.
    Бот ответит строго по тексту документов со ссылками на источники.
    """)

    chatbot = gr.Chatbot(
        height=500,
        # show_copy_button=True,
        # bubble_full_width=False,
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Например: Какие управляющие параметры используются в модели роста опухоли?",
            scale=8,
            container=False,
            show_label=False
        )
        submit_btn = gr.Button("Отправить", variant="primary", scale=1)

    clear_btn = gr.ClearButton([msg, chatbot], value="Очистить чат")

    # Логика
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    # Точка входа: запуск веб-интерфейса
    print("Запуск веб-интерфейса")
    demo.launch(theme="soft")