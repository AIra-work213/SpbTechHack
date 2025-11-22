from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from prepare_data import prepare_data
from parsing import extract_visible_text
from extract_data import extract_text


all_urls = ["https://gu.spb.ru/knowledge-base"]
client = QdrantClient("http://localhost:6333")

collection_name = "knowledge_database"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE
        )
    )
if __name__ == "__main__":
    try:
        for url in all_urls:
            client.upsert(collection_name=collection_name, points=prepare_data(extract_visible_text(url)))
    except Exception as e:
        print(f"Ошибка при добавлении данных: {e}")
    
    try:
        query = "Как зарегистрироваться?"
        query_vector = extract_text(query)
        if not query_vector:
            raise ValueError("Не удалось получить вектор запроса")
        result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=12,
            with_payload=True
        )
        print(result)
    except Exception as e:
        print(f"Ошибка при поиске данных: {e}")