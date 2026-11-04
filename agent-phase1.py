import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="qwen/qwen3-8b")

parser = StrOutputParser()
chain = llm | parser

messages = []
messages.append(HumanMessage("Your a helpful FinTech Assistant explain stock market in simple words"))

while True:
    user_input = input("Enter your message: ")
    messages.append(HumanMessage(user_input))

    response = chain.invoke(messages)
    print("Response:", response)
    messages.append(AIMessage(response))
