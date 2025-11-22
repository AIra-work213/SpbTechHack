from fastapi import FastAPI

app = FastAPI()

@app.post("/query")
async def query():
    