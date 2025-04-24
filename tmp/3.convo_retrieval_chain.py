from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
)
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

llm = ChatOllama(model="llama3.2", temperature=0.8, num_predict=256)

raw_documents = TextLoader("../data/state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
documents = text_splitter.split_documents(raw_documents)

embedding = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(documents, embedding=embedding)

query = "What did the president say about ukraine"
docs = db.similarity_search(query)

print(docs[0].page_content)
print(docs[1].page_content)
