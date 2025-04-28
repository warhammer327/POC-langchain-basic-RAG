from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

# Initialize LLM
llm = ChatOllama(model="llama3.2", temperature=0.7)

# Load documents from the link for our knowledge base
print("Loading and processing documents...")
docs = WebBaseLoader("https://www.sevensix.co.jp/topics/iqom_invention-award/").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(docs)

# print(f"Created {len(documents)} document chunks")
# for doc in documents:
#    print(f"{doc}\n")
# print("#################################")

# Create embedding model and vector store
embedding = OllamaEmbeddings(model="nomic-embed-text")
# Chromadb provides in-memory vector storage
vectorstore = Chroma.from_documents(documents, embedding=embedding)
# Create retriever with appropriate k value based on document count
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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

document_chain = create_stuff_documents_chain(llm, qa_prompt)
conversation_retrieval_chain = create_retrieval_chain(
    history_aware_retriever, document_chain
)


# Function to run a conversation with the chain
def chat_with_documents(chat_history=None):
    if chat_history is None:
        chat_history = []

    while True:
        user_input = input("\nYour question (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break

        # Process the query with the conversation chain
        response = conversation_retrieval_chain.invoke(
            {"chat_history": chat_history, "input": user_input}
        )

        # Print the answer
        print("\nAI:", response["answer"])

        # Update chat history with this exchange
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response["answer"]))

        # Optional: Print retrieved documents (for debugging)
        # print("\nRetrieved context:")
        # for i, doc in enumerate(response["context"], 1):
        #     print(f"Document {i}: {doc.page_content[:150]}...")


# Example 1: Starting a new conversation
def example_1():
    print("\n===== EXAMPLE 1: New Conversation =====")
    # Initial query about LangSmith
    chat_history = []
    query = "what is iQoM?"

    print(f"User: {query}")
    response = conversation_retrieval_chain.invoke(
        {"chat_history": chat_history, "input": query}
    )
    print(f"AI: {response['answer']}")

    # Update chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["answer"]))

    # Follow-up question
    follow_up = "what wavelength are generated from it?"
    print(f"\nUser: {follow_up}")
    response = conversation_retrieval_chain.invoke(
        {"chat_history": chat_history, "input": follow_up}
    )
    print(f"AI: {response['answer']}")

    # Update chat history
    chat_history.append(HumanMessage(content=follow_up))
    chat_history.append(AIMessage(content=response["answer"]))

    # Second follow-up question referring to previous context
    second_follow_up = "how many wavelength does it have ?"
    print(f"\nUser: {second_follow_up}")
    response = conversation_retrieval_chain.invoke(
        {"chat_history": chat_history, "input": second_follow_up}
    )
    print(f"AI: {response['answer']}")

    chat_history.append(HumanMessage(content=second_follow_up))
    chat_history.append(AIMessage(content=response["answer"]))

    third_follow_up = "how it has achieved lower cost?"
    print(f"\nUser: {third_follow_up}")
    response = conversation_retrieval_chain.invoke(
        {"chat_history": chat_history, "input": third_follow_up}
    )
    print(f"AI: {response['answer']}")


# Example 2: Testing with ambiguous follow-up questions
def example_2():
    print("\n===== EXAMPLE 2: Ambiguous Follow-up Questions =====")
    chat_history = [
        HumanMessage(content="What is LangSmith?"),
        AIMessage(
            content="LangSmith is a platform for developing and monitoring LLM applications."
        ),
    ]

    follow_up = "How does it work?"
    print(f"User: {follow_up}")
    response = conversation_retrieval_chain.invoke(
        {"chat_history": chat_history, "input": follow_up}
    )
    print(f"AI: {response['answer']}")

    # Update chat history
    chat_history.append(HumanMessage(content=follow_up))
    chat_history.append(AIMessage(content=response["answer"]))

    # Very ambiguous follow-up that relies on conversation history
    ambiguous_query = "Can you tell me more about that?"
    print(f"\nUser: {ambiguous_query}")
    response = conversation_retrieval_chain.invoke(
        {"chat_history": chat_history, "input": ambiguous_query}
    )
    print(f"AI: {response['answer']}")


# Run the examples
if __name__ == "__main__":
    print("Conversation Retrieval Chain with Persistent Context")
    print("===================================================")

    # Run the examples
    # example_1()
    # example_2()

    # Interactive mode
    print("\n===== INTERACTIVE MODE =====")
    print("Start a conversation (type 'exit' to quit)")
    chat_with_documents()
