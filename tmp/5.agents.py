from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults

llm = ChatOllama(model="llama3.2", temperature=0.7)
search = TavilySearchResults()
