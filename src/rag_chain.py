import json
import re
from typing import List, Dict, Any, Optional
from vector_store import VectorStore
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from build_index import LocalEmbedder
import time

class RAGChain:
    """

    """
    def __init__(self, collection_name,  llm_model="Qwen/Qwen2.5-1.5B-Instruct", qdrant_url="http://localhost:6333"):
        """

        :param collection_name:
        :param qdrant_url:
        :param embedding_model:
        :param llm_config:
        """
        self.collection_name = collection_name
        self.vector_store = VectorStore(collection_name, url=qdrant_url)
        self.embedding_model = LocalEmbedder()
        self.llm = pipeline("text-generation", model=llm_model)


    def ask_rag(self, question, filters=None, top_k=5):
        """

        :param question:
        :param filters:
        :return:
        """
        query_vector = self.embedding_model.encode([question])[0].tolist()

        chunks = self.vector_store.search_vectors(
            collection_name=self.collection_name,
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
                "authors": chunk.payload.get("authors", ""),
                "year": chunk.payload.get("year"),
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
            generated_text = raw_output[0]['generated_text']
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
        return f"""Ты — помощник-исследователь. Твоя задача: ответить на вопрос, используя ТОЛЬКО предоставленный контекст.
        Если ответа в тексте нет, напиши "Информация не найдена".
    
        ### КОНТЕКСТ:
        {context}
    
        ### ВОПРОС:
        {question}
    
        ### ИНСТРУКЦИЯ:
        Ответь в формате JSON:
        {{
          "answer": "текст ответа со ссылками на источники в формате [1], [2]",
          "citations": [
            {{"id": 1, "source": "название/автор", "page": "номер или null"}}
          ]
        }}
        JSON:"""


    def _calculate_confidence(self, chunks):
        if not chunks:
            return "low"
        avg_score = sum(chunk.score for chunk in chunks) / len(chunks)
        if avg_score > 0.8:
            return "high"
        elif avg_score > 0.5:
            return "medium"
        return "low"
