from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = ChatOllama(model="llama3.2", temperature=0.3)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_edge(START, "chatbot")
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for val in event.values():
            print("Assistant:", val["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["exit"]:
            break
        stream_graph_updates(user_input)
    except Exception as e:
        user_input = "what do you know about langgraph?"
        print(e)
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
