from pandasai.llm.base import LLM
from pandasai import SmartDataframe
from app.core.parser.response_parser import PandasDataFrame
from litellm import completion
import os
import logging

class LMStudio(LLM):
    def __init__(self, model: str, base_url: str = None, api_key: str = None, **kwargs):
        self.model = model
        self.api_base = base_url or os.getenv("LITELLM_BASE_URL")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")

        if not self.api_base:
            raise ValueError("El parámetro 'base_url' o la variable de entorno LITELLM_BASE_URL es obligatorio.")

    def chat(self, messages, **kwargs):
        try:
            response = completion(
                model=self.model,
                messages=messages,
                api_base=self.api_base,
                api_key=self.api_key,
                **kwargs
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Error comunicando con LLM ({self.model}): {e}", exc_info=True)
            raise


class MyFlexibleLLM:
    def __init__(
        self,
        model: str = "hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct",
        name: str = None,
        description: str = None,
        config=None,
        base_url: str = None,
        api_key: str = None,
        **kwargs
    ):
        self.llm = LMStudio(model=model, base_url=base_url, api_key=api_key)

        self.name = name
        self.description = description or "Rows of data"
        self.config = config or {
            "enforce_privacy": False,
            "llm": self.llm,
            "verbose": True,
            "response_parser": PandasDataFrame,
        }

    def chat(self, df, prompt, config=None):
        df = SmartDataframe(
            df, name=self.name, description=self.description, config=config or self.config
        )
        return df.chat(prompt)

# Alias exportable
MyOpenAI = MyFlexibleLLM
