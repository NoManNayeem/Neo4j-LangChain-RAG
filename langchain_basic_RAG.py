import os
from config import OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

# 1) Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 2) Load & split documents
loader = PyPDFDirectoryLoader("data")
raw_docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = splitter.split_documents(raw_docs)

# 3) Index into Neo4jVector
db = Neo4jVector.from_documents(
    docs,
    embeddings,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# 4) Set up LLM & prompt template
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context (each chunk prefixed by its source path):
{context}

Answer (include a source citation at the end in square brackets, e.g. [Source: filename.pdf]):"""
    ),
])

# 5) Helper to format retrieved docs
def format_docs(docs_with_scores):
    texts = []
    for doc, score, *rest in docs_with_scores:
        page = getattr(doc, "page_content", str(doc))
        source = doc.metadata.get("source", "unknown")
        texts.append(f"[Source: {source}]\n{page}")
    return "\n\n".join(texts)

# 6) Retrieve, format, build prompt, invoke LLM, and parse
query = "What is Dynamic Bayesian Stackelberg Games?"
hits = db.similarity_search_with_score(query, k=3)
formatted_context = format_docs(hits)

chat_prompt = prompt.format_prompt(question=query, context=formatted_context)
messages = chat_prompt.to_messages()

llm_response = llm.invoke(messages)
answer_text = StrOutputParser().parse(llm_response.content)

# 7) Print answer and sources
unique_sources = []
for doc, *_ in hits:
    src = doc.metadata.get("source", "unknown")
    if src not in unique_sources:
        unique_sources.append(src)

print("\n=== ANSWER ===")
print(answer_text)

print("\n=== SOURCES ===")
for src in unique_sources:
    print(f"- {src}")
