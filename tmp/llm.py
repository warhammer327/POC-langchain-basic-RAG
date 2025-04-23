from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(
    model="llama3.2",
    temperature=0.8,
    num_predict=256,
    # other params ...
)

question = "tell me about japan"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "give two line answer."),
        ("user", "{input}"),
    ]
)

output_parsers = StrOutputParser()


chain = prompt | llm | output_parsers

print("Welcome to the console chatbot! Type 'exit' to quit.")

while True:
    user_input = input("\nYou:")

    if user_input.lower() in ["exit"]:
        break

    try:
        response = chain.invoke({"input": user_input})
        print(f"\nBot: {response}")
    except Exception as e:
        print(f"\nBot: {e}")
