from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains.history_aware_retriever import create_history_aware_retriever


from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document


# Initialize LLM
llm = ChatOllama(model="llama3.2", temperature=0.8, num_predict=256)

# Load and split documents properly
docs = WebBaseLoader(
    "https://docs.smith.langchain.com/prompt_engineering/quickstarts/quickstart_ui"
).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# Check number of documents
print(f"Number of documents: {len(documents)}")

# Create embedding model and vector store
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents, embedding=embedding)

# Create a prompt template that explicitly instructs to use the context
prompt = ChatPromptTemplate.from_template(
    """
You are an assistant that answers questions based ONLY on the context provided below.
If the information isn't in the context, say "I don't have that information in the provided context."

Context:
{context}

Question: {input}

Answer:
"""
)

# Create document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create retriever with appropriate k value based on document count
retriever = vectorstore.as_retriever(search_kwargs={"k": min(4, len(documents))})

# Create retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Run the chain and print both retrieved documents and the answer
response = retrieval_chain.invoke({"input": "how to test a prompt?"})
print("\n##########")
print("RETRIEVED DOCUMENTS:")
for i, doc in enumerate(response["context"]):
    print(f"Document {i + 1}:\n{doc.page_content[:200]}...\n")

print("##########")
print("ANSWER:")
print(response["answer"])
