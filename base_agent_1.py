# base_agent_1.py

import warnings
# Suppress the “API key must be provided when using hosted LangSmith API” warning
warnings.filterwarnings(
    "ignore",
    message="API key must be provided when using hosted LangSmith API"
)

from config import OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

# ─── 1. LLM ────────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4"
)

# ─── 2. Embeddings ────────────────────────────────────────────────────────
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY
)

# ─── 3. Neo4j Graph ───────────────────────────────────────────────────────
from langchain_neo4j import Neo4jGraph

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# ─── 4. Prompt + Tool ─────────────────────────────────────────────────────
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a movie expert providing information about movies."),
    ("human", "{input}")
])

movie_chat = chat_prompt | llm | StrOutputParser()

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general movie Q&A not covered by other tools",
        func=movie_chat.invoke
    )
]

# ─── 5. Conversation Memory ───────────────────────────────────────────────
from langchain_neo4j import Neo4jChatMessageHistory

def get_memory(session_id: str):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# ─── 6. Agent Setup ───────────────────────────────────────────────────────
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory

agent_prompt = hub.pull("hwchase17/react-chat")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=agent_prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# ─── 7. Generate Response ─────────────────────────────────────────────────
def generate_response(user_input: str, session_id: str = "default") -> str:
    result = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": session_id}}
    )
    return result["output"]

# ─── 8. Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    reply = generate_response("Hello, who directed Inception?", session_id="session1")
    print("Agent ⇢", reply)
