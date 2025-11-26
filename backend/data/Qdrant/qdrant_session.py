from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from backend.data.Qdrant.parsing import extract_visible_text
import asyncio
import json
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct
import uuid


class KnowledgeBase:
    def __init__(self, url="http://localhost:6333", collection_name="knowledge_base_ru_small"):
        self.client = QdrantClient(url)
        self.collection_name = collection_name
        self.all_urls = ["https://gu.spb.ru/knowledge-base"]
        self.model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
        self._loaded = False
        
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
        else:
            try:
                count_result = self.client.count(collection_name=self.collection_name, exact=False)
                self._loaded = count_result.count > 0
            except Exception:
                self._loaded = False

    async def load(self):
        if self._loaded:
            return
        try:
            for url in self.all_urls:
                self.client.upsert(
                    collection_name=self.collection_name, 
                    points=await self.prepare_data(extract_visible_text(url))
                )
            self._loaded = True
        except Exception as e:
            print(f"Ошибка при добавлении данных: {e}")

    async def search_vector(self, query):
        try:
            query_vector = await self.extract_text(query)
            result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=12,
                    with_payload=True
                )
            return result
        except Exception as e:
            print(f"Ошибка при поиске данных: {e}")
            return []
        
    async def prepare_data(self, data):
        my_data = json.loads(data)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200
        )

        points = []
        for sample in my_data:
            chunks = text_splitter.split_text(sample["text"])
            for i, chunk in enumerate(chunks):
                # e5 models expect 'passage: ' prefix for documents
                vector = self.model.embed_documents([f"passage: {chunk}"])[0]
                
                chunk_payload = sample.copy()
                chunk_payload["text"] = chunk
                chunk_payload["chunk_index"] = i
                
                points.append(PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{sample['url']}_{i}")),
                    vector=vector,
                    payload=chunk_payload
                ))
        return points

    async def extract_text(self, query):
        try:
            return self.model.embed_query(f"query: {query}")
        except Exception as e:
            print(f"Ошибка при извлечении текста: {e}")
            return None




async def main():
    kb = KnowledgeBase()
    if not kb.client.collection_exists(kb.collection_name):
        await kb.load()
    query = "Как зарегистрироваться?"
    result = await kb.search_vector(query)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())