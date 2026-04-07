import qdrant_client
from qdrant_client import models
from typing import List, Optional, Dict, Any
from qdrant_client.http.models import Distance
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config import QDRANT_API_KEY

class VectorStore:
    """
    Клиент для работы с Qdrant.
    """

    def __init__(
            self,
            collection_name: str,
            url: str = "http://localhost:6333"
    ) -> None:
        """
        Конструктор класса.
        """
        self.client = qdrant_client.QdrantClient(
            url=url,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
        )
        self.collection_name = collection_name

    def create_collection(self, vector_size: int) -> None:
        """
        Создаёт коллекцию, если её нет.
        """
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

    def upload_vectors(
            self,
            vectors: List[List[float]],
            payloads: List[Dict[str, Any]],
            ids: List[Any]
    ) -> None:
        """
        Загружает векторы с метаданными.
        """
        points = [models.PointStruct(id=id, vector=vector, payload=payload)
                  for id, vector, payload in zip(ids, vectors, payloads)]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search_vectors(
            self,
            query_vector: List[float],
            filter: Optional[models.Filter] = None,
            top_k: int = 5
    ) -> Any:
        """
        Поиск похожих векторов.
        """
        # return self.client.search(
        #     collection_name=self.collection_name,
        #     query_vector=query_vector,
        #     query_filter=filter,
        #     limit=top_k
        # )
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=filter,
            limit=top_k,
            with_payload=True
        )
        return results.points

if __name__ == "__main__":
    # Тестовые данные
    COLLECTION = "test_collection"
    VECTOR_SIZE = 4  # Маленький вектор для теста
    URL = "http://localhost:6333"

    # Инициализация
    store = VectorStore(collection_name=COLLECTION, url=URL)
    store.create_collection(vector_size=VECTOR_SIZE)

    # 1) Тест загрузки
    test_ids = [101, 102, 103]
    test_vectors = [
        [0.1, 0.2, 0.3, 0.4],
        [0.9, 0.8, 0.7, 0.6],
        [0.1, 0.1, 0.1, 0.1],
    ]
    test_payloads = [
        {"text": "первый чанк", "tag": "a"},
        {"text": "второй чанк", "tag": "b"},
        {"text": "третий чанк", "tag": "a"},
    ]

    print(f" Загрузка {len(test_ids)} векторов.")
    store.upload_vectors(
        vectors=test_vectors,
        payloads=test_payloads,
        ids=test_ids
    )

    # 2) Тест поиска
    query = [0.15, 0.25, 0.35, 0.45]
    print(f" Поиск по запросу {query}.")

    results = store.search_vectors(query_vector=query, top_k=2)

    print(f"Найдено результатов: {len(results)}")
    for r in results:
        print(f"   ID: {r.id} | Score: {r.score:.4f} | Payload: {r.payload}")

    # 3) Тест с фильтром
    print("Поиск с фильтром (tag == 'b').")
    from qdrant_client import models

    filter_query = models.Filter(
        must=[models.FieldCondition(key="tag", match=models.MatchValue(value="b"))]
    )

    results_filtered = store.search_vectors(
        query_vector=query,
        filter=filter_query,
        top_k=5
    )

    for r in results_filtered:
        print(f"   ID: {r.id} | Score: {r.score:.4f} | Payload: {r.payload}")
