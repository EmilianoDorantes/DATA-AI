import pandas as pd
from google.cloud import storage
import io

def read_data_with_dtype(path_to_data, dtype=None):
    """
    Lee datos desde Google Cloud Storage o sistema de archivos local
    path_to_data: puede ser una ruta local o una ruta de GCS (gs://bucket-name/path/to/file)
    """
    # Verificar si es una ruta de GCS
    if path_to_data.startswith('gs://'):
        # Extraer bucket y blob name
        bucket_name = path_to_data.split('/')[2]
        blob_name = '/'.join(path_to_data.split('/')[3:])
        
        # Inicializar cliente de GCS
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Descargar contenido a memoria
        content = blob.download_as_bytes()
        file_obj = io.BytesIO(content)
        
        # Obtener extensión del archivo
        file_extension = blob_name.split('.')[-1].lower()
    else:
        file_extension = path_to_data.split('.')[-1].lower()
        file_obj = path_to_data

    # Procesar según el tipo de archivo
    if file_extension == "json":
        if dtype is None:
            return pd.read_json(file_obj, lines=True)
        else:
            return pd.read_json(file_obj, lines=True, dtype=dtype)
    elif file_extension == "csv":
        if dtype is not None:
            return pd.read_csv(file_obj, dtype=dtype)
        else:
            return pd.read_csv(file_obj)
    elif file_extension in ["xls", "xlsx"]:
        if dtype is None:
            return pd.read_excel(file_obj)
        else:
            return pd.read_excel(file_obj, dtype=dtype)
    elif file_extension in ["snappy", "parquet"]:
        if dtype is None:
            return pd.read_parquet(file_obj)
        else:
            return pd.read_parquet(file_obj, dtype=dtype)
    else:
        raise ValueError(f"Formato de archivo no soportado: {file_extension}")
