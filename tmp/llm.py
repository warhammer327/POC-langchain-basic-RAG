from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(
    model="llama3.2",
    temperature=0.8,
    num_predict=256,
    # other params ...
)

template = """You are a grammar transformation expert. Convert the given sentence from {input_form} voice to {output_form} voice.
Maintain the original meaning and only output the transformed sentence."""
human_message = "{text}"

chat_prompt = ChatPromptTemplate.from_messages(
    [("system", template), ("user", human_message)]
)

output_parsers = StrOutputParser()


chain = chat_prompt | llm | output_parsers

print("Welcome to the console chatbot! Type 'exit' to quit.")


while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit"]:
        break

    try:
        response = chain.invoke(
            {"input_form": "active", "output_form": "passive", "text": user_input}
        )
        print(f"\nBot: {response}")
    except Exception as e:
        print(f"\nBot: {e}")
