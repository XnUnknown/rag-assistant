
import json
import math
import re
from pathlib import Path
from collections import Counter

import ollama

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.json"

EMBED_MODEL = "nomic-embed-text"

VECTOR_CANDIDATES = 30
KEYWORD_CANDIDATES = 30
FINAL_CHUNKS = 8
RRF_K = 60
CONTEXT_CHAR_LIMIT = 24000

def load_index():
    if not INDEX_FILE.exists():
        raise FileNotFoundError("index.json not found. Run embed_kb.py first.")

    with open(INDEX_FILE, "r", encoding="utf-8") as file:
        return json.load(file)

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

def embed_query(query):
    result = ollama.embed(model=EMBED_MODEL, input=query)
    return result["embeddings"][0]

def cosine_similarity(a, b):
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))

    if not mag_a or not mag_b:
        return 0.0

    return dot / (mag_a * mag_b)

def vector_search(query_vector, chunks):
    results = []

    for item in chunks:
        vector = item.get("vec")

        if not vector:
            continue

        score = cosine_similarity(query_vector, vector)

        results.append((item, score))

    return sorted(results, key=lambda x: x[1], reverse=True)[:VECTOR_CANDIDATES]

def keyword_search(query, chunks):
    query_terms = Counter(tokenize(query))
    documents = [Counter(tokenize(item.get("text", ""))) for item in chunks]

    if not query_terms or not documents:
        return []

    total_docs = len(documents)
    avg_length = sum(sum(doc.values()) for doc in documents) / total_docs
    avg_length = max(avg_length, 1)

    doc_freq = Counter()
    for doc in documents:
        for term in query_terms:
            if doc[term] > 0:
                doc_freq[term] += 1

    results = []

    for item, doc in zip(chunks, documents):
        doc_length = sum(doc.values())
        score = 0.0

        for term, query_frequency in query_terms.items():
            frequency = doc[term]

            if not frequency:
                continue

            idf = math.log(
                1 + (total_docs - doc_freq[term] + 0.5) /
                (doc_freq[term] + 0.5)
            )

            k1 = 1.5
            b = 0.75

            denominator = frequency + k1 * (
                1 - b + b * doc_length / avg_length
            )

            score += idf * (
                frequency * (k1 + 1) / denominator
            ) * query_frequency

        if score > 0:
            results.append((item, score))

    return sorted(results, key=lambda x: x[1], reverse=True)[:KEYWORD_CANDIDATES]

def chunk_key(item):
    return (item.get("doc"), item.get("chunk"))

def hybrid_search(query, chunks):
    query_vector = embed_query(query)

    vector_results = vector_search(query_vector, chunks)
    keyword_results = keyword_search(query, chunks)

    fused = {}

    for ranking in (vector_results, keyword_results):
        for rank, (item, score) in enumerate(ranking, start=1):
            key = chunk_key(item)

            if key not in fused:
                fused[key] = {
                    "item": item,
                    "rrf": 0.0,
                    "vector_score": None,
                    "keyword_score": None
                }

            fused[key]["rrf"] += 1 / (RRF_K + rank)

            if ranking is vector_results:
                fused[key]["vector_score"] = score
            else:
                fused[key]["keyword_score"] = score

    ranked = sorted(
        fused.values(),
        key=lambda x: x["rrf"],
        reverse=True
    )

    return ranked

def add_neighboring_chunks(selected, chunks):
    by_doc = {}

    for item in chunks:
        by_doc.setdefault(item["doc"], {})[item["chunk"]] = item

    expanded = {}

    for result in selected:
        item = result["item"]
        doc = item["doc"]
        number = item["chunk"]

        for nearby_number in (number - 1, number, number + 1):
            neighbor = by_doc.get(doc, {}).get(nearby_number)
            if neighbor:
                key = chunk_key(neighbor)
                expanded[key] = neighbor
    return sorted(
        expanded.values(),
        key=lambda x: (x["doc"], x["chunk"])
    )
def rag_search(query, k=FINAL_CHUNKS):
    chunks = load_index()
    if not chunks:
        return []
    ranked = hybrid_search(query, chunks)
    selected = ranked[:max(1, int(k))]
    expanded = add_neighboring_chunks(selected, chunks)
    return [
        {
            **item,
            "score": next(
                (
                    result["rrf"]
                    for result in selected
                    if chunk_key(result["item"]) == chunk_key(item)
                ),
                0.0
            )
        }
        for item in expanded
    ]
def build_context(results):
    if not results:
        return "No relevant information was found in the knowledge base."
    sections = []
    total_chars = 0
    for item in results:
        section = (
            f"[Source: {item['doc']} | "
            f"Pages: {item.get('page_start', '?')}-"
            f"{item.get('page_end', '?')} | "
            f"Chunk: {item['chunk']}]\n"
            f"{item['text']}"
        )

        if total_chars + len(section) > CONTEXT_CHAR_LIMIT:
            break

        sections.append(section)
        total_chars += len(section)

    return "\n\n---\n\n".join(sections)
