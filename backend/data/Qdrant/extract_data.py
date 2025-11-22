from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
url = "https://foundation-models.api.cloud.ru/v1"

client = OpenAI(
    api_key=api_key,
    base_url=url
)

def extract_text(query):
    try:
        return client.embeddings.create(
            model="Qwen/Qwen3-Embedding-0.6B",
            input=[query]
        ).data[0].embedding
    except Exception as e:
        print(f"Ошибка при векторном представлении запроса: {e}")
        return None