import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from langgraph_cli.constants import SUPABASE_URL
from vector_store import search_documents
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

# Load environment variables
load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
tools = [search_documents]
llm_with_tools = llm.bind_tools(tools)

def call_llm(state: MessagesState):
    """Call LLM"""

    system_prompt = SystemMessage(
        content=
            "You MUST use the search_documents tool for course material questions. "
            "CRITICAL: Do NOT use standard Markdown formatting. Do not use # headers, > blockquotes, or **bold**. "
            "Format your responses as plain, conversational text. Use bullet points if necessary."
    )

    message_to_send = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(message_to_send)

    return {"messages": [response]}

# Supabase configurations
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0
}
pool = ConnectionPool(conninfo=SUPABASE_DB_URL, max_size=10, kwargs=connection_kwargs)
memory = PostgresSaver(pool)
memory.setup()

# Build the agent
builder = StateGraph(MessagesState)

# Nodes
builder.add_node("chatbot", call_llm)
builder.add_node("tools", ToolNode(tools))

# Edges
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

# Compile the graph
agent = builder.compile(checkpointer=memory)


