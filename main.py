import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="qwen/qwen3-8b")

parser = StrOutputParser()

prompt_template = """
Your a helpful assistant who responds to the user with genz slang and helps the user: {user_input}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

genz_chain = prompt | llm | parser

user_message = input("Enter message: ")
response = genz_chain.invoke({"user_input": user_message})
print(response)
