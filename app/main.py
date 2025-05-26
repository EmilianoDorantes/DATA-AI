from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config.config import create_agent_instance

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

agent = create_agent_instance()


@app.post("/chat")
async def get_my_chat(prompt: str, file_path: str):
    return agent.chat(prompt, file_path=file_path)