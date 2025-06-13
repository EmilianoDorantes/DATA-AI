from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from app.core.config.config import create_agent_instance
import requests
import tempfile
import os

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    file_path: str   # Puede ser URL o path local
    api_key: str
    base_url: str

def download_file_from_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    extension = url.split('.')[-1].split('?')[0]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{extension}') as tmp_file:
        tmp_file.write(response.content)
        return tmp_file.name

@app.post("/chat")
async def get_my_chat(request: ChatRequest):
    file_path = request.file_path
    temp_file_created = False
    if file_path.startswith("http://") or file_path.startswith("https://"):
        try:
            file_path = download_file_from_url(file_path)
            temp_file_created = True
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No se pudo descargar el archivo: {e}"
            )

    try:
        agent = create_agent_instance(
            path_to_data=file_path,
            api_key=request.api_key,
            base_url=request.base_url
        )
        result = agent.chat(request.prompt)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la interacción con el LLM: {e}"
        ) from e
    finally:
        # Limpia archivo temporal si fue descargado
        if temp_file_created and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
