from typing import List, Dict, Any, Optional
from vector_store import VectorStore
from transformers import pipeline
from build_index import LocalEmbedder
import sys
from pathlib import Path
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config import QDRANT_COLLECTION, QDRANT_URL, LLM_MODEL, EMBEDDING_MODEL, DEVICE

class RAGChain:
    """
    Класс объединяет векторный поиск в Qdrant и генерацию ответов с помощью LLM.
    """
    def __init__(self, collection_name=QDRANT_COLLECTION,  llm_model=LLM_MODEL, qdrant_url=QDRANT_URL):
        """
        Инициализация RAG цепочки.
        """
        self.collection_name = collection_name
        self.vector_store = VectorStore(collection_name, url=qdrant_url)
        self.embedding_model = LocalEmbedder(model=EMBEDDING_MODEL)
        self.llm = pipeline(
            "text-generation",
            model=llm_model,
            device=DEVICE,
            model_kwargs={
            "dtype": torch.float16,
            "low_cpu_mem_usage": True
        })

    def ask_rag(self, question, filters=None, top_k=5):
        """
        Основной метод RAG: задать вопрос и получить ответ с источниками.
        """
        query_vector = self.embedding_model.encode([question])[0]

        chunks = self.vector_store.search_vectors(
           # collection_name=self.collection_name,
            query_vector=query_vector,
            filter=filters,
            top_k=top_k
        )

        if not chunks:
            return {
                "answer": "Не нашёл контекста в базе знаний",
                "citations": [],
                "confidence": "low"
            }

        citations = [
            {
                "source": chunk.payload.get("source_path", ""),
                "authors": chunk.payload.get("authors", "") or "Автор неизвестен",
                "year": chunk.payload.get("year") or "г.н.",
                "journal": chunk.payload.get("journal", ""),
                "page": chunk.payload.get("page"),
                "doi": chunk.payload.get("doi", ""),
                "snippet": (chunk.payload.get("text", "")[:200] + "...")
            }
            for chunk in chunks
        ]

        context_text = "\n".join([f"[{i + 1}] {chunk.payload.get('text', '')}" for i, chunk in enumerate(chunks)])
        prompt = self._build_prompt(context_text, question)

        raw_output = self.llm(prompt, max_new_tokens=512, do_sample=False)

        if isinstance(raw_output, list):
            generated_text = raw_output[0]["generated_text"]
            answer_text = generated_text.replace(prompt, "").strip()
        else:
            answer_text = raw_output

        confidence = self._calculate_confidence(chunks)

        return {
            "answer": answer_text,
            "citations": citations,
            "confidence": confidence
        }

    def _build_prompt(self, context, question):
        """
        Формирует промпт для LLM.
        """
        # return f"""Ты — помощник-исследователь. Твоя задача: ответить на вопрос, используя ТОЛЬКО предоставленный контекст.
        # Если ответа в тексте нет, напиши "Информация не найдена".
        #
        # ### КОНТЕКСТ:
        # {context}
        #
        # ### ВОПРОС:
        # {question}
        #
        # ### ОТВЕТ (кратко, по делу):"""
        return f"""Ты — научный ассистент. Ответь кратко на вопрос, используя ТОЛЬКО текст ниже.
        Если ответа нет, так и скажи.

        КОНТЕКСТ:
        {context}

        ВОПРОС: {question}

        ОТВЕТ:"""

    def _calculate_confidence(self, chunks):
        """
        Вычисляет уровень уверенности в ответе на основе оценок релевантности.
        """
        if not chunks:
            return "low"
        avg_score = sum(chunk.score for chunk in chunks) / len(chunks)
        if avg_score > 0.8:
            return "high"
        elif avg_score > 0.5:
            return "medium"
        return "low"

if __name__ == "__main__":
    # Точка входа для интерактивного чата с RAG системой.
    rag = RAGChain(QDRANT_COLLECTION)
    print("RAG-чат запущен (введите /exit для выхода)")

    while True:
        q = input("Введите запрос: ").strip()
        if q in ["/exit", "quit", "выход"]:
            break
        if not q:
            continue

        res = rag.ask_rag(q)
        print(f"\n{res['answer']}")
        if res["citations"]:
            print("\nИсточники:")
            for i, c in enumerate(res["citations"], 1):
                print(f"{i}. {c['authors']} ({c['year']}) — {c['snippet'][:100]}...")
        print(f"[{res['confidence']}]")
