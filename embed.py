
import json
import re
from pathlib import Path

import fitz
import ollama

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_FILE = BASE_DIR / "index.json"

EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 650
CHUNK_OVERLAP = 100

def clean_text(text):
    text = re.sub(r"-\s*\n\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()

def chunk_pages(pages):
    chunks = []
    words = []
    page_numbers = []

    for page_number, text in pages:
        page_words = clean_text(text).split()
        words.extend(page_words)
        page_numbers.extend([page_number] * len(page_words))

    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_words = words[start:end]

        if chunk_words:
            chunks.append({
                "text": " ".join(chunk_words),
                "page_start": page_numbers[start],
                "page_end": page_numbers[end - 1]
            })

        if end == len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks

def embed(text):
    result = ollama.embed(model=EMBED_MODEL, input=text)
    return result["embeddings"][0]

def process_pdf(pdf_path):
    print(f"Reading: {pdf_path.name}")

    with fitz.open(pdf_path) as document:
        pages = [
            (number + 1, page.get_text("text"))
            for number, page in enumerate(document)
        ]

    chunks = chunk_pages(pages)
    items = []

    for number, chunk in enumerate(chunks):
        print(f"Embedding {number + 1}/{len(chunks)}", end="\r")

        try:
            vector = embed(chunk["text"])
        except Exception as error:
            print(f"\nEmbedding error: {error}")
            continue

        items.append({
            "doc": pdf_path.name,
            "chunk": number,
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "text": chunk["text"],
            "vec": vector
        })

    print(f"\nFinished: {pdf_path.name} ({len(items)} chunks)")
    return items

def main():
    DOCS_DIR.mkdir(exist_ok=True)
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in docs.")
        return

    all_items = []

    for pdf in pdf_files:
        all_items.extend(process_pdf(pdf))

    with open(INDEX_FILE, "w", encoding="utf-8") as file:
        json.dump(all_items, file, ensure_ascii=False)

    print(f"\nIndex created: {len(all_items)} chunks")
    print(f"Saved to: {INDEX_FILE}")

if __name__ == "__main__":
    main()
