import re
import warnings
from typing import Annotated, Dict, Optional, List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import create_retriever_tool
from typing_extensions import TypedDict
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_weaviate import WeaviateVectorStore
import weaviate
from weaviate.exceptions import WeaviateConnectionError

from pydantic import BaseModel, Field

warnings.filterwarnings(
    "ignore",
    message=".*Accessing the 'model_fields' attribute on the instance is deprecated.*",
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ollama._types")
warnings.filterwarnings(
    "ignore", category=PendingDeprecationWarning, module="ollama._types"
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    query_assessment: Optional[Dict]
    search_results: Optional[List[Dict]]


class QueryAssessment(BaseModel):
    """Assessment of user query quality."""

    is_well_formed: bool = Field(
        description="Whether the query is grammatically correct"
    )
    improved_query: str = Field(description="Improved version of the query if needed")
    needs_web_search: bool = Field(
        description="Whether a web search would help answer this query"
    )
    explanation: str = Field(description="Brief explanation of the assessment")


query_assessment_parser = PydanticOutputParser(pydantic_object=QueryAssessment)

query_assessment_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You assess user queries to determine if they're grammatically correct and clear. 
    If a query is vague, poorly structured, or unclear, improve it to be more specific and effective.
    Also determine if the query would benefit from web search results to provide a complete response.
    
    Output your assessment in the following format:
    {format_instructions}
    """,
        )
    ]
).partial(format_instructions=query_assessment_parser.get_format_instructions)


def assess_query(state: State):
    """Evaluate if the query is well formed or not"""
    user_query = state["messages"][-1].content
    state["query"] = user_query

    # Run the assessment
    assessment_result = llm.with_structured_output(QueryAssessment).invoke(
        query_assessment_prompt.format(query=user_query)
    )

    print(f"Query Assessment: {assessment_result}")

    # Update the message with the final query
    state["messages"][-1].content = user_query

    return state


def chatbot(state: State):
    """Generate response to user query"""
    # Get the final query from state
    final_query = state["messages"][-1].content

    # Generate response
    response = llm.invoke(final_query)

    return {"messages": [{"role": "assistant", "content": response.content}]}


# Initialize components
llm = ChatOllama(model="llama3.2", temperature=0.7)
embedding = OllamaEmbeddings(model="nomic-embed-text")

# Set up the graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("assess_query", assess_query)
graph_builder.add_node("chatbot", chatbot)

# Define edges
graph_builder.add_edge(START, "assess_query")
graph_builder.add_edge("assess_query", "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    """Stream the graph execution"""
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for val in event.values():
            if "messages" in val:
                for message in val["messages"]:
                    if message["role"] == "assistant":
                        print("Assistant:", message.content)


# Main interaction loop
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        stream_graph_updates(user_input)
    except Exception as e:
        print(f"Error: {e}")
        break

print("Goodbye!")
