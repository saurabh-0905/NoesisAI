# ingest.py
# ------------------------------------------------------------------
# Run this file whenever you want to teach Noesis AI new documents.
# It reads every file in the "documents/" folder, breaks each one
# into small overlapping chunks, and stores those chunks in ChromaDB
# so the research agent can find them later.
#
# Usage:  python ingest.py
# ------------------------------------------------------------------

import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from tools import get_chroma_collection

DOCS_FOLDER = "documents"


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_file_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return read_pdf(path)
    elif path.lower().endswith((".txt", ".md")):
        return read_txt(path)
    else:
        return ""  # unsupported file type, skip it


def ingest_all():
    if not os.path.isdir(DOCS_FOLDER):
        print(f"No '{DOCS_FOLDER}' folder found. Create it and add PDFs/TXT files.")
        return

    files = [f for f in os.listdir(DOCS_FOLDER) if not f.startswith(".")]
    if not files:
        print(f"'{DOCS_FOLDER}' is empty - nothing to ingest.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    collection = get_chroma_collection()
    total_chunks = 0

    for filename in files:
        path = os.path.join(DOCS_FOLDER, filename)
        text = load_file_text(path)

        if not text.strip():
            print(f"  skipped (unsupported or empty): {filename}")
            continue

        chunks = splitter.split_text(text)

        # Give each chunk a unique, stable ID so re-running ingest
        # doesn't create duplicates for the same file.
        ids = [f"{filename}-{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in chunks]

        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        total_chunks += len(chunks)
        print(f"  ingested: {filename}  ({len(chunks)} chunks)")

    print(f"\nDone. {total_chunks} chunks stored in ChromaDB at '{config.CHROMA_DB_PATH}'.")


if __name__ == "__main__":
    ingest_all()
