# RAG Assistant

A local **Retrieval-Augmented Generation (RAG)** assistant for answering
technical questions over a corpus of PDF documents. It embeds PDF chunks
with a local Ollama embedding model, then serves a tool-calling chat
assistant that retrieves relevant context (hybrid semantic + keyword search)
before answering, with citations.

## Setup

1. Install [Ollama](https://ollama.com/) and pull the required models:

   ```bash
   ollama pull nomic-embed-text
   ollama pull lfm3-omni:2b   # or any other chat model you configure
   ```

2. Create a Python virtual environment and install dependencies:

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Put your PDFs in the `docs/` directory.

4. Build the embedding index:

   ```bash
   python embed.py
   ```

5. Run the assistant:

   ```bash
   python assistant.py
   ```

   Type questions into the prompt; type `exit` or `quit` to leave.

## How it works

- `embed.py` — extracts text from every PDF in `docs/`, chunks it with a
  small overlap, embeds each chunk with `nomic-embed-text` via Ollama, and
  writes `index.json`.
- `rag.py` — loads `index.json`, performs hybrid retrieval:
  - vector search (cosine similarity on embeddings)
  - keyword search (BM25-style scoring)
  - fused with Reciprocal Rank Fusion (RRF)
  - expands selected hits with neighboring chunks for better context
- `assistant.py` — an Ollama chat agent with a `search_knowledge` tool. For
  technical topics it calls the retriever first, then streams a final answer
  grounded in the retrieved context and cites documents and pages.

Technical answers are refused-by-policy when no supporting chunks are found:
the assistant reports that the documents don't contain the answer rather than
inventing one.

## Configuration

Tune constants at the top of each script:

- `assistant.py`: `MODEL`, `DEFAULT_K`, `MAX_TOOL_ROUNDS`
- `rag.py`: `EMBED_MODEL`, `VECTOR_CANDIDATES`, `KEYWORD_CANDIDATES`,
  `FINAL_CHUNKS`, `RRF_K`, `CONTEXT_CHAR_LIMIT`
- `embed.py`: `EMBED_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`

## Notes

- `index.json` and `venv/` are gitignored — regenerate the index locally
  with `embed.py`.
- Default configuration targets a local Ollama server at
  `http://localhost:11434`. Remote/hosted Ollama works via the commented
  snippet in `assistant.py`.
