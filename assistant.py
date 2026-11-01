
import json
import os
import time

from pathlib import Path

import ollama

from rag import rag_search, build_context

# ---------------- CONFIGURATION ----------------

MAX_TOOL_ROUNDS = 5
DEFAULT_K = os.getenv("DEFAULT_K", 8)

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

MODEL =  os.getenv("MODEL", "gemma4:31b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else None
client = ollama.Client(host=OLLAMA_HOST, headers=headers)
# ---------------- CONVERSATION ----------------

messages = [
    {
        "role": "system",
        "content": """
You are a technical research assistant connected to a PDF knowledge base.

RULES:
- For technical questions, research papers, AI architectures,
  machine learning, neural networks and technical concepts,
  you MUST call search_knowledge before answering.
- Do not answer technical questions from memory without searching.
- If the first search is insufficient, search again using
  alternative technical terminology.
- Combine relevant retrieved passages.
- Base your answer on retrieved documents.
- Cite document names and page numbers.
- Never invent facts, numbers or citations.
- If the documents do not contain the answer, say so clearly.
- Explain concepts in detail.
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
                "Search the local PDF knowledge base using semantic "
                "and keyword retrieval. MUST be used for technical questions."
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
            result = search_knowledge(
                arguments["query"],
                arguments.get("k", DEFAULT_K)
            )
            print("[TOOL COMPLETED]")
            return result

        except Exception as error:
            print(f"[RAG ERROR] {error}")
            return {"error": str(error), "found": False}

    return {"error": f"Unknown tool: {name}"}

# ---------------- ASSISTANT ----------------

def ask(question):
    messages.append({
        "role": "user",
        "content": question
    })

    for round_no in range(MAX_TOOL_ROUNDS):
        print(f"\n[MODEL ROUND {round_no + 1}]")

        # Tool-calling stage: non-streaming
        response = client.chat(
            model=MODEL,
            messages=messages,
            tools=tools,
            stream=False
        )

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

            print()

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
