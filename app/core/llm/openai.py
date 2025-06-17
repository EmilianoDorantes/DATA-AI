from pandasai import SmartDataframe
from pandasai.llm import OpenAI

from app.core.parser.response_parser import PandasDataFrame

# URL base personalizada de OpenAI
OPENAI_API_BASE_URL = "https://devilproxy-798669397793.europe-west1.run.app/v1/chat/completions"

class MyOpenAI(OpenAI):

    def __init__(
        self,
        api_token,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        name: str = None,
        description: str = None,
        config=None,
        **kwargs
    ):
        super().__init__(api_token, base_url=OPENAI_API_BASE_URL, **kwargs)
        self.llm = OpenAI(api_token=api_token, model=model, base_url=OPENAI_API_BASE_URL)

        if name is None:
            name = "DataFrame"

        if description is None:
            description = "Rows of data"

        if config is None:
            config = {
                "enforce_privacy": False,
                "llm": self.llm,
                "verbose": True,
                "response_parser": PandasDataFrame,
            }

        self.name = name
        self.description = description
        self.config = config

    def chat(self, df, prompt, config):
        df = SmartDataframe(
            df, name=self.name, description=self.description, config=config
        )
        return df.chat(prompt)
