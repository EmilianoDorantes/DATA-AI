import json
import logging

from fastapi.responses import JSONResponse
from pandasai.responses.response_parser import ResponseParser


class PandasDataFrame(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def parse(self, result):
        logging.info(f"[PARSER] Recibido: {result}")

        fallback_msg = "No pude interpretar tu solicitud. Intenta reformularla o revisa el archivo."

        try:
            if not isinstance(result, dict):
                logging.error(f"[PARSER] Se esperaba dict, se recibió {type(result)}")
                return JSONResponse(
                    {"type": "error", "value": str(result)},
                    status_code=400
                )

            response_type = result.get("type")
            response_value = result.get("value")

            if response_type in ["dataframe", "plot", "string", "number"]:
                return JSONResponse(
                    {"type": response_type, "value": response_value},
                    status_code=200
                )

            logging.warning(f"[PARSER] Tipo inesperado: {response_type}")
            return JSONResponse(
                {"type": "error", "value": fallback_msg},
                status_code=400
            )

        except Exception as e:
            logging.exception(f"[PARSER] Error interno: {e}")
            return JSONResponse(
                {"type": "error", "value": f"Error interno del parser: {e}"},
                status_code=500
            )
