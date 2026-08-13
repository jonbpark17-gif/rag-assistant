# RAG Document Q&A Assistant

A retrieval-augmented generation (RAG) app that answers questions over a set of
documents and cites its sources.

## How it works
1. Documents are split into chunks and embedded into a Chroma vector store (locally).
2. A question retrieves the most relevant chunks.
3. Those chunks are passed as context to Claude, which answers using only that context.

## Stack
Python, LangChain, ChromaDB, Anthropic API, Streamlit

## Run it
1. `pip install -r requirements.txt`
2. Add your key to a `.env` file: `ANTHROPIC_API_KEY=your-key`
3. `python rag.py`
