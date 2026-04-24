from typing import List, Dict, Any, Optional
from vector_store import VectorStore
from transformers import pipeline, AutoTokenizer
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
    def __init__(
            self,
            collection_name: str = QDRANT_COLLECTION,
            llm_model: str = LLM_MODEL,
            qdrant_url: str = QDRANT_URL
    ) -> None:
        """
        Инициализация RAG цепочки.
        """
        self.collection_name = collection_name
        self.vector_store = VectorStore(collection_name, url=qdrant_url)
        self.embedding_model = LocalEmbedder(model=EMBEDDING_MODEL)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        except:
            self.tokenizer = None

        self.llm = pipeline(
            "text-generation",
            model=llm_model,
            device=DEVICE,
            model_kwargs={
            "dtype": torch.float16,
            "low_cpu_mem_usage": True
        })

    def ask_rag(
            self,
            question: str,
            filters: Optional[Any] = None,
            top_k: int = 3
    ) -> Dict[str, Any]:
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

        raw_output = self.llm(
            prompt,
            max_new_tokens=256,
            do_sample=False,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else 0
        )

        if isinstance(raw_output, list):
            generated_text = raw_output[0]["generated_text"]
            # answer_text = generated_text.replace(prompt, "").strip()
        else:
            generated_text = raw_output

        answer_text = self._clean_answer(generated_text)

        confidence = self._calculate_confidence(chunks)

        return {
            "answer": answer_text,
            "citations": citations,
            "confidence": confidence
        }

    def _clean_answer(self, text: str) -> str:
        """
        Удаляет хвосты генерации, где модель начинает придумывать новые вопросы.
        """
        if not text:
            return ""

        stop_markers = [
            "\n\nВОПРОС:",
            "\nВОПРОС:",
            "\n\nQuestion:",
            "\nQuestion:",
            "\n\n### ВОПРОС:",
            "<|user|>",
            "<|start_header_id|>user"
        ]

        min_index = len(text)
        for marker in stop_markers:
            idx = text.find(marker)
            if idx != -1 and idx < min_index:
                min_index = idx

        if min_index < len(text):
            text = text[:min_index]

        text = text.strip()

        if text.upper().startswith("ОТВЕТ:"):
            text = text[len("ОТВЕТ:"):].strip()

        return text

    def _build_prompt(self, context: str, question: str) -> str:
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
        # return f"""Ты — научный ассистент. Ответь кратко на вопрос, используя ТОЛЬКО текст ниже.
        # Если ответа нет, так и скажи.
        #
        # КОНТЕКСТ:
        # {context}
        #
        # ВОПРОС: {question}
        #
        # ОТВЕТ:"""
        #
        prompt = f"""Ты — научный ассистент. Ответь кратко на вопрос, используя ТОЛЬКО текст ниже.
        Если ответа нет, так и скажи: "Информация не найдена".

        КОНТЕКСТ:
        {context}

        ВОПРОС: {question}

        ОТВЕТ:"""
        return prompt

    def _calculate_confidence(self, chunks: List[Any]) -> str:
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
