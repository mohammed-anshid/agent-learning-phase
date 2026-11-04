import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatGroq(
    model="qwen/qwen3-8b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    max_retries=2
)

@tool
def get_current_time() -> str:
    """Get the current UTC date and time"""
    return f"Current UTC time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"

tools = [get_current_time]


SYSTEM_PROMPT = (
    "You are an intelligent, helpful agent powered by Groq and LangGraph. "
    "You have tools for date/time lookup. "
    "Always invoke appropriate tools when relevant and provide concise, accurate answers."
)

# Export the compiled StateGraph
# This symbol `graph` is referenced in `langgraph.json` as `./agent.py:graph`
graph = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
)
