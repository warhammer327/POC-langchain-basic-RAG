from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel


class WeatherResponse(BaseModel):
    conditions: str
    recommendation: str


model = init_chat_model("ollama:llama3.2")


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


checkpointer = InMemorySaver()

agent = create_react_agent(
    model="ollama:llama3.2",
    tools=[get_weather],
    prompt="You are a assistant",
    checkpointer=checkpointer,
    response_format=WeatherResponse,
)

config = {"configurable": {"thread_id": "1"}}

res = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}, config
)
print(res["structured_response"])
