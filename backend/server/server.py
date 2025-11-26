from fastapi import FastAPI
import uvicorn
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession
import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel



sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.agents.system import AgentSystem

load_dotenv()


class QueryStructure(BaseModel):
    query: str


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://foundation-models.api.cloud.ru/v1"
)

app = FastAPI()
agent_system = AgentSystem()

@app.post("/query")
async def query(query_str: QueryStructure):
    query = query_str.query
    
    final_message = await agent_system.graph_invoke(query)

            
    return {"response": final_message}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)