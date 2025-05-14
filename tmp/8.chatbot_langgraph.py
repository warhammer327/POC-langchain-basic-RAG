import warnings
import re
from typing import cast

from langchain.tools.retriever import create_retriever_tool
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

import weaviate
from langchain_weaviate import WeaviateVectorStore
from weaviate.exceptions import WeaviateConnectionError

from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import convert_to_messages

warnings.filterwarnings(
    "ignore",
    message=".*Accessing the 'model_fields' attribute on the instance is deprecated.*",
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ollama._types")
warnings.filterwarnings(
    "ignore", category=PendingDeprecationWarning, module="ollama._types"
)


class State(MessagesState):
    documents: list[str]


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


llm = init_chat_model(model="llama3.2", model_provider="ollama", temperature=0.3)
embedding = OllamaEmbeddings(model="nomic-embed-text")


def load_content():
    urls = ["https://www.sevensix.co.jp/topics/iqom_invention-award/"]
    docs = [WebBaseLoader(url).load() for url in urls]
    print("=====Web content======")
    print(docs)
    return docs


def split_documents(docs):
    docs[0][0].page_content.strip()[:1000]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(docs_list)
    doc_splits[0].page_content.strip()
    return doc_splits


def create_vectordb_client():
    client = weaviate.connect_to_local(
        host="localhost",
        port=8087,
    )
    return client


def close_vectordb_client(client):
    client.close()


def create_vectorstore(client):
    vectorstore = WeaviateVectorStore(
        client=client,
        index_name="reward_hack",
        text_key="text",
        embedding=embedding,
    )
    return vectorstore


def make_document_list(doc_splits_list):
    document_list = []
    chunk_id = 1
    for doc in doc_splits_list:
        if not re.search(r"\n{1,}", doc.page_content):
            document_dict = {
                "chunk_id": chunk_id,
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            document_list.append(document_dict)
            chunk_id += 1
    return document_list


def insert_data_to_vectordb(client):
    class_scheme_name = "test_prods_1"
    collection = client.collections.get(class_scheme_name)

    docs = load_content()
    doc_splits_list = split_documents(docs)
    document_list = make_document_list(doc_splits_list)

    print("======Creating product class schema======")
    if not client.collections.exists(class_scheme_name):
        client.collections.create(name=class_scheme_name)

    collection = client.collections.get(class_scheme_name)
    print("======Inserting documents======")

    try:
        is_empty = collection.aggregate.over_all(total_count=True)
        # print(f"-> {document_list}\n")
        if is_empty.total_count == 0:
            print("======Insertion on going======")
            for doc in document_list:
                vector = embedding.embed_documents([doc["page_content"]])[0]
                collection.data.insert(
                    properties={
                        "text": doc["page_content"],
                        "source": doc["metadata"].get("source", ""),
                        "chunk_id": doc["chunk_id"],
                    },
                    vector=vector,
                )
        print("======Insertion completed======")
        print(f"======Read from {class_scheme_name} collection======")
        collection = client.collections.get(class_scheme_name).query.fetch_objects(
            limit=100
        )
        for i, item in enumerate(collection.objects):
            print(f"{item}\n\n")
    except Exception as e:
        print(f"Error inserting document: {e}")


def init_retriever_tool(vectorstore):
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 10,
        }
    )
    print("======Testing retriever======")
    try:
        results = retriever.invoke("what is iqom?")
        print(f"Retrieved {len(results)} documents for test query 'iqom'")
        for i, doc in enumerate(results):
            print(f"Test result {i}: {doc.page_content[:100]}...")
    except Exception as e:
        print(f"Error testing retriever: {e}")
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_device_data",
        "Search and return information about devices and products including IQOM, FRUSH, and SPECTOR",
    )
    return retriever_tool


client = create_vectordb_client()
insert_data_to_vectordb(client)
vectorstore = create_vectorstore(client)
retriever_tool = init_retriever_tool(vectorstore)
print("=" * 50)
print("=" * 50)
print("=" * 50)


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


input = cast(
    MessagesState,
    {
        "messages": [
            {
                "role": "user",
                "content": "What does iqom does?",
            }
        ]
    },
)

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = llm.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


input = cast(
    MessagesState,
    {
        "messages": convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "what is the use of iqom device ?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_device_data",
                            "args": {"query": "What does iqom device do?"},
                        }
                    ],
                },
                {"role": "tool", "content": "meow", "tool_call_id": "1"},
            ]
        )
    },
)

grade_documents(input)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke([{"role": "user", "content": prompt}])
    print("*" * 50)
    print(messages)
    print("*" * 50)
    print(response.content)
    print("*" * 50)
    return {"messages": [{"role": "user", "content": response.content}]}


input = cast(
    MessagesState,
    {
        "messages": convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "tell me about iqom device",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_blog_posts",
                            "args": {"query": "what does iqom device do?"},
                        }
                    ],
                },
                {"role": "tool", "content": "meow", "tool_call_id": "1"},
            ]
        )
    },
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


input = cast(
    MessagesState,
    {
        "messages": convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "what is iqom device ?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_blog_posts",
                            "args": {"query": "what does iqom device do?"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "iQoM series is an all-fiber-based compact ultrashort pulse laser module. Picosecond pulses of 1040 nm or 1064 nm wavelength are generated by the 976 nm CW laser input to the iQoM.",
                    "tool_call_id": "1",
                },
            ]
        )
    },
)

response = generate_answer(input)
# response["messages"][-1].pretty_print()


workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()


for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "tell me about iqom device",
            }
        ]
    }
):
    for node, update in chunk.items():
        print("Update from node", node)
        # update["messages"][-1].pretty_print()
        print("%" * 50)
        print(update["messages"][-1])
        print("%" * 50)
        print("\n\n")

close_vectordb_client(client)
