from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import requests
import os
import atexit
import logging

from pandasai import Agent
from pandasai.connectors import PandasConnector
from pandasai.connectors.pandas import PandasConnectorConfig
from app.core.llm.openai import LMStudio
from app.core.llm.openai import MyOpenAI
from app.core.parser.response_parser import PandasDataFrame
from app.core.services.reader import read_data_with_dtype
from app.core.services.helpers import get_env
from fastapi import HTTPException, status

# Configurar el logging
# Cargar variables de entorno desde .env
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo para el cuerpo de la solicitud
class ChatRequest(BaseModel):
    prompt: str
    file_path: str

# Registro de archivos temporales
temp_files = []

def register_temp_file(path):
    temp_files.append(path)

@atexit.register
def cleanup_temp_files():
    for path in temp_files:
        try:
            os.remove(path)
            print(f"[CLEANUP] Archivo temporal eliminado: {path}")
        except Exception as e:
            print(f"[CLEANUP ERROR] No se pudo eliminar {path}: {e}")

# Función para crear agente dinámicamente con cualquier dataset
def create_agent_with_data(data) -> Agent:
    connector = PandasConnector(PandasConnectorConfig(original_df=data))
    # Usar LMStudio directamente ya que hereda de LLM
    llm_instance = LMStudio(
        base_url="https://api.runpod.ai/v2/a2auhmx8h7iu3z/openai/v1",
        model="hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct")
    return Agent(
        connector,
        config={
            "llm": llm_instance,
            "response_parser": PandasDataFrame,
            "custom_head": data.head(25),
        },
    )
# Descarga archivo desde una URL remota y lo guarda temporalmente
def download_file_from_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()

    extension = url.split('.')[-1].split('?')[0]  # Maneja URLs con query strings
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{extension}') as tmp_file:
        tmp_file.write(response.content)
        temp_path = tmp_file.name
        register_temp_file(temp_path)
        return temp_path

# Endpoint principal
@app.post("/chat")
async def get_my_chat(request: ChatRequest):
    original_file_path = request.file_path
    processed_file_path = original_file_path

    if processed_file_path.startswith("http://") or processed_file_path.startswith("https://"):
        try:
            processed_file_path = download_file_from_url(processed_file_path)
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error al descargar el archivo desde la URL: {e}")

    try:
        data = read_data_with_dtype(processed_file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No se encontró el archivo: {original_file_path}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al leer el archivo: {e}")

    try:
        agent = create_agent_with_data(data)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al crear el agente de PandasAI: {e}")

    try:
        logger.info(f"Sending prompt to agent.chat(): {request.prompt}") # Log the prompt being sent
        result = agent.chat(request.prompt)
        logger.info(f"Raw PandasAI result: {result}") # Log the raw result
        return result
    except Exception as e:
        # Catch potential errors during LLM interaction
        logger.error(f"Error during LLM interaction: {e}", exc_info=True) # Log the error with traceback
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error durante la interacción con el LLM: {e}") from e


###Esta API fue diseñada por Emiliano Dorantes, con la finalidad de hacerla compatible con OPEN WEBUI
