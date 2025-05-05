import warnings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_weaviate import WeaviateVectorStore

import re
import weaviate
from weaviate.exceptions import WeaviateConnectionError

warnings.filterwarnings(
    "ignore",
    message=".*Accessing the 'model_fields' attribute on the instance is deprecated.*",
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ollama._types")
warnings.filterwarnings(
    "ignore", category=PendingDeprecationWarning, module="ollama._types"
)


# Initialize LLM
llm = ChatOllama(model="llama3.2", temperature=0.3)

# Load documents from the link for our knowledge base
print("======Loading and processing documents======")
docs = WebBaseLoader("https://www.sevensix.co.jp/topics/iqom_invention-award/").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(docs)

document_list = []
chunk_id = 1
for doc in documents:
    if not re.search(r"\n{1,}", doc.page_content):
        document_dict = {
            "chunk_id": chunk_id,
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        document_list.append(document_dict)
        chunk_id += 1

# Create embedding model and vector store
print("======Choosing embedding======")
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = None
print("======Connecting to weaviate======")
try:
    client = weaviate.connect_to_local(
        host="localhost",
        port=8087,
    )
    print("======Creating product class schema======")
    if not client.collections.exists("Product"):
        client.collections.create(name="Product")

    collection = client.collections.get("Product")
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
    except Exception as e:
        print(f"Error inserting document: {e}")

    try:
        print("======Setup vector store======")
        vectorstore = WeaviateVectorStore(
            client=client,
            index_name="Product",
            text_key="text",
            embedding=embedding,
        )
    except Exception as e:
        print(f"{e}")
except WeaviateConnectionError as e:
    print(f"Error connecting to weaviate: {e}")


if not vectorstore:
    raise RuntimeError("vectorstore initialization failed")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
# Create a history-aware retriever that generates search queries based on conversation
retriever_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up information relevant to the conversation",
        ),
    ]
)

print("======Making history aware retriever======")
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, retriever_prompt
)

# Create the conversation chain with context and history
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions based on the retrieved context. "
            "Keep the answer short. "
            "If the information isn't in the context, acknowledge what you don't know.\n\n"
            "Context: {context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

print("======Making document chain======")
document_chain = create_stuff_documents_chain(llm, qa_prompt)
conversation_retrieval_chain = create_retrieval_chain(
    history_aware_retriever, document_chain
)


# Function to run a conversation with the chain
def chat_with_documents(chat_history=None):
    if chat_history is None:
        chat_history = []

    # Add cache dictionary to store query-response pairs
    cache = {}

    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == "exit":
            break

        # Create a cache key based on chat history and current input
        # We need to convert messages to strings for hashing
        history_str = str([(msg.type, msg.content) for msg in chat_history])
        cache_key = f"{history_str}_{user_input}"

        print("-->AI:", end="", flush=True)

        # Check if response is in cache
        if cache_key in cache:
            response = cache[cache_key]
            print(" (cached response)", end="", flush=True)
        else:
            # Process the query with the conversation chain using invoke
            response_dict = conversation_retrieval_chain.invoke(
                {"chat_history": chat_history, "input": user_input}
            )

            # Extract the answer
            response = response_dict["answer"]

            # Store in cache
            cache[cache_key] = response

        # Print the response
        print(response, flush=True)
        print(f"\n->{cache}\n")
        # Update chat history with this exchange
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

    try:
        if client and client.is_ready():
            client.close()
            print("\nWeaviate connection closed.")
    except Exception as e:
        print(f"\nError closing Weaviate connection: {e}")


# Run the examples
if __name__ == "__main__":
    print("\nConversation Retrieval Chain with Persistent Context")
    # print("===================================================")

    # Run the examples
    # example_2()

    # start_time = time.time()
    # example_1()
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"\nExecution time: {execution_time:.2f} seconds")
    print("\n===== INTERACTIVE MODE =====")
    print("Start a conversation (type 'exit' to quit)")
    chat_with_documents()
