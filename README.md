# Neo4j + LangChain RAG Demo

This project demonstrates how to use Neo4j as a vector store for Retrieval-Augmented Generation (RAG) with LangChain and OpenAI.

## Overview
- **Neo4jVector**: Store and query vector embeddings in Neo4j.
- **LangChain**: Handle document loading, splitting, and chaining with LLMs.
- **OpenAI**: Generate embeddings and answer questions.

## Prerequisites
- Python 3.8+
- Neo4j instance (with authentication credentials)
- OpenAI API key

## Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/NoManNayeem/Neo4j-LangChain-RAG.git
   cd Neo4j-LangChain-RAG
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Set environment variables in `config.py`:
   ```python
   OPENAI_API_KEY = "your-openai-key"
   NEO4J_URI = "bolt://localhost:7687"
   NEO4J_USERNAME = "neo4j"
   NEO4J_PASSWORD = "password"
   ```
2. Place your PDF files in the `data/` directory.
3. Run the demo script:
   ```bash
   uv run langchain_basic_RAG.py
   ```
4. Ask a question and view the generated answer with source citations.

