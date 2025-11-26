from mcp.server.fastmcp import FastMCP
import sys
import os

# Add backend directory to sys.path to allow imports from data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.data.Qdrant.qdrant_session import KnowledgeBase
import asyncio

vector_store = KnowledgeBase()

app = FastMCP(
    "mytools",
)

# @app.tool()
# async def get_memory():
#     pass

@app.tool()
async def search_qdrant(query: str) -> str:
    """Поиск релеватной информации в базе знаний Санкт-Петербурга. Используй для каждого запроса пользователя"""
    import json
    result = await vector_store.search_vector(query)
    
    # Extract text and URLs from results
    extracted_data = []
    urls = set()
    
    if hasattr(result, 'points'):
        for point in result.points:
            payload = point.payload if hasattr(point, 'payload') else {}
            text = payload.get('text', '')
            url = payload.get('url', '')
            if url:
                urls.add(url)
            extracted_data.append({
                'text': text,
                'url': url
            })
    
    return json.dumps({
        'results': extracted_data,
        'urls': list(urls)
    }, ensure_ascii=False)

# @app.tool()
# async def web_search(query: str) -> str:
#     pass

if __name__ == "__main__":
    asyncio.run(vector_store.load())
    app.run()

