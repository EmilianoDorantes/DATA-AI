from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.core.config.config import create_agent_instance

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/chat")
async def get_my_chat(prompt: str, data_url: str = Query(...)):
    agent = create_agent_instance(data_url)
    return agent.chat(prompt)
