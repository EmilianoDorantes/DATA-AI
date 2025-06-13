from pandasai import Agent
from pandasai.connectors import PandasConnector
from pandasai.connectors.pandas import PandasConnectorConfig

from app.core.llm.openai import MyOpenAI
from app.core.parser.response_parser import PandasDataFrame
from app.core.services.reader import read_data_with_dtype

def create_agent_instance(path_to_data, api_key, base_url) -> Agent:
    data = read_data_with_dtype(path_to_data)
    connector = PandasConnector(
        PandasConnectorConfig(original_df=data),
    )
    data_custom_head = data.head(25)
    llm = MyOpenAI(
        api_key=api_key,
        base_url=base_url
    )
    return Agent(
        connector,
        config={
            "llm": llm,
            "response_parser": PandasDataFrame,
            "custom_head": data_custom_head,
        },
    )
