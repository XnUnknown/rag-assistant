
import json
import os
import time

from pathlib import Path

import ollama

from rag import rag_search, build_context

# ---------------- CONFIGURATION ----------------

MODEL = os.getenv("MODEL", "gemma4:31b")
# MODEL = "lfm3-omni:2b"
MAX_TOOL_ROUNDS = 5
DEFAULT_K = int(os.getenv("DEFAULT_K", 8))

BASE_DIR = Path(__file__).resolve().parent


def load_env():
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else None
client = ollama.Client(host=OLLAMA_HOST, headers=headers)
# ---------------- CONVERSATION ----------------

messages = [
    {
        "role": "system",
        "content": """
You are an expert Neo4j Database and Cypher technical assistant connected to the official Neo4j documentation knowledge base.

RULES:
- For questions about Neo4j, Cypher query syntax, graph data modeling,
  database operations, configuration (neo4j.conf), Python driver APIs,
  status codes, performance tuning, and technical concepts,
  you MUST call search_knowledge before answering.
- Do not answer Neo4j or technical questions from memory without searching.
- If the first search is insufficient, search again using alternative
  technical terminology (e.g., exact Cypher clauses, config keys, error codes).
- Provide clear, syntactically valid Cypher queries, code snippets, or configuration examples when appropriate.
- Base your answers strictly on the retrieved documentation passages.
- Cite the exact document names and page numbers (e.g., "neo4j-cypher-manual-25.pdf, pages 45-46").
- Never invent clauses, procedures, configuration settings, or status codes.
- If the documentation does not contain the answer, say so clearly.
- Explain concepts and query mechanics clearly.
- For greetings and general conversation, tool usage is unnecessary.
"""
    }
]

# ---------------- TOOL DEFINITION ----------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search the official Neo4j documentation knowledge base (Cypher manual, "
                "Operations manual, Python driver, status codes) using semantic and "
                "keyword retrieval. MUST be used for Neo4j and technical questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Technical topic or detailed search query."
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of main search results."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ---------------- RAG SEARCH ----------------

def search_knowledge(query, k=DEFAULT_K):
    print(f"\n[RAG] Searching: {query}")

    results = rag_search(query, k=int(k))
    context = build_context(results)

    print(f"[RAG] Retrieved {len(results)} chunks")

    return {
        "found": bool(results),
        "context": context,
        "sources": [
            {
                "document": item["doc"],
                "pages": (
                    f"{item.get('page_start', '?')}-"
                    f"{item.get('page_end', '?')}"
                ),
                "chunk": item["chunk"]
            }
            for item in results
        ]
    }

# ---------------- TOOL EXECUTION ----------------

def execute_tool(tool_call):
    name = tool_call.function.name
    arguments = tool_call.function.arguments

    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    print(f"\n[TOOL REQUESTED] {name}")
    print(f"[TOOL ARGUMENTS] {arguments}")

    if name == "search_knowledge":
        try:
            search_started = time.perf_counter()
            result = search_knowledge(
                arguments["query"],
                arguments.get("k", DEFAULT_K)
            )
            search_elapsed = time.perf_counter() - search_started
            print(f"[TOOL COMPLETED] in {search_elapsed:.2f}s")
            return result

        except Exception as error:
            print(f"[RAG ERROR] {error}")
            return {"error": str(error), "found": False}

    return {"error": f"Unknown tool: {name}"}

# ---------------- ASSISTANT ----------------

def ask(question):
    total_started = time.perf_counter()

    messages.append({
        "role": "user",
        "content": question
    })

    for round_no in range(MAX_TOOL_ROUNDS):
        print(f"\n[MODEL ROUND {round_no + 1}]")

        # Tool-calling stage: non-streaming
        llm_started = time.perf_counter()
        response = client.chat(
            model=MODEL,
            messages=messages,
            tools=tools,
            stream=False
        )
        llm_elapsed = time.perf_counter() - llm_started
        print(f"[MODEL ROUND {round_no + 1}] LLM latency: {llm_elapsed:.2f}s")

        assistant_message = response.message
        tool_calls = assistant_message.tool_calls or []

        # No tool call: stream the final response
        if not tool_calls:
            content = assistant_message.content or ""

            if not content:
                print("[WARNING] Model returned an empty response.")
                return "The model returned an empty response."

            messages.append({
                "role": "assistant",
                "content": content
            })

            print("\nAssistant: ", end="", flush=True)

            # Stream final answer
            stream_started = time.perf_counter()
            stream = client.chat(
                model=MODEL,
                messages=messages[:-1] + [
                    {"role": "user", "content": question}
                ],
                stream=True
            )

            final_answer = ""

            for chunk in stream:
                if chunk.message.content:
                    text = chunk.message.content
                    final_answer += text
                    print(text, end="", flush=True)

            stream_elapsed = time.perf_counter() - stream_started
            total_elapsed = time.perf_counter() - total_started
            print()
            print(f"[Latency] final stream: {stream_elapsed:.2f}s | total: {total_elapsed:.2f}s")

            # Keep the actual streamed answer in history
            messages[-1] = {
                "role": "assistant",
                "content": final_answer
            }

            return final_answer

        # Save assistant tool-call message
        messages.append(
            assistant_message.model_dump(exclude_none=True)
        )

        # Execute requested tools
        for tool_call in tool_calls:
            result = execute_tool(tool_call)

            messages.append({
                "role": "tool",
                "tool_name": tool_call.function.name,
                "content": json.dumps(
                    result,
                    ensure_ascii=False
                )
            })

    return "Maximum tool-call limit reached."

# ---------------- MAIN ----------------

def main():
    print(f"Assistant ready: {MODEL}")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            ask(question)

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break

        except Exception as error:
            print(f"\n[ERROR] {type(error).__name__}: {error}")

# ---------------- START ----------------

if __name__ == "__main__":
    main()
