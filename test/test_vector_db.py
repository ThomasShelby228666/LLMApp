from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

client = QdrantClient(url="http://localhost:6333")

# Удаляем существующую коллекцию (если есть)
try:
    client.delete_collection("articles")
    print("Старая коллекция удалена")
except:
    pass

# Создаём коллекцию, если её нет (384 = длина эмбеддинга all-MiniLM-L6-v2)
client.create_collection(
    collection_name="articles",
    vectors_config=rest.VectorParams(size=384, distance=rest.Distance.COSINE)
)

sentences = [
    "Сьешь ещё этих мягких франзцузcких булок и выпей ещё чаю", # id = 0
    "В этом предложении все буквы алфавита", # id = 1
    "С кайфом покушал  сегодня" # id = 2
]

model = SentenceTransformer("all-MiniLM-L6-v2")

vectors = model.encode(sentences, normalize_embeddings=True)

payloads = [
    {"tag": "food"},
    {"tag": "language"},
    {"tag": "food"}
]

points=[
        rest.PointStruct(id=i, vector=v.tolist(), payload=payloads[i])
        for i, v in enumerate(vectors)
    ]

client.upsert(
    collection_name="articles",
    points=points
)

query_vec = model.encode(
    "После шаурмы охота в тулает", normalize_embeddings=True
)

hits = client.query_points(
    collection_name="articles",
    query=query_vec.tolist(),
    limit=2
)

print("\nВсе результаты:")
for hit in hits.points:
    print(f"score={hit.score:.3f}  ->  {sentences[hit.id]}")

hits_filtered = client.query_points(
    collection_name="articles",
    query=query_vec.tolist(),
    limit=2,
    query_filter=rest.Filter(
        must=[rest.FieldCondition(key="tag", match=rest.MatchValue(value="food"))]
    )
)

print("\nРезультаты с фильтрацией:")
for hit in hits_filtered.points:
    print(f"score={hit.score:.3f}  ->  {sentences[hit.id]}")