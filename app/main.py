from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.core.config.config import create_agent_instance, create_agent_instance_with_path

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/chat")
async def get_my_chat(prompt: str, data_url: str = Query(None)):
    if data_url:
        agent = create_agent_instance_with_path(data_url)
    else:
        agent = create_agent_instance()
    return agent.chat(prompt)
