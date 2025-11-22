from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from qdrant_client.models import PointStruct
import uuid
from openai import OpenAI
from dotenv import load_dotenv

import os

load_dotenv()

api_key = os.getenv("API_KEY")
url = "https://foundation-models.api.cloud.ru/v1"

client = OpenAI(
    api_key=api_key,
    base_url=url
)



def prepare_data(data):
    my_data = json.loads(data)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    points = []
    for sample in my_data:
        response = client.embeddings.create(
            model="Qwen/Qwen3-Embedding-0.6B",
            input=[sample["text"]]
        )
        points.append(PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, sample["url"])),
            vector=response.data[0].embedding,
            payload=sample
        ))
    
    return points