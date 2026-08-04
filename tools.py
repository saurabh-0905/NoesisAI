# tools.py
# ------------------------------------------------------------------
# This file has two simple jobs:
#   1. Look something up in OUR OWN uploaded documents (ChromaDB)
#   2. Look something up on the LIVE WEB (Tavily)
# Every other file just calls these two functions - they don't need
# to know HOW the lookup happens.
# ------------------------------------------------------------------

import os
import chromadb
from chromadb.utils import embedding_functions
from tavily import TavilyClient
from dotenv import load_dotenv

import config

load_dotenv()  # reads GROQ_API_KEY / TAVILY_API_KEY from a .env file


def get_chroma_collection():
    """
    Connects to our local vector database on disk and returns the
    collection where all ingested document chunks live.
    Creates it automatically the first time it's called.
    """
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

    # This turns text into embeddings automatically using a model
    # that runs on your own machine (no API call, no cost).
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embed_fn,
    )
    return collection


def get_ingested_sources():
    """
    Returns a sorted list of unique filenames currently stored in
    ChromaDB. Used by the sidebar to show WHAT has been ingested,
    not just how many chunks.
    """
    collection = get_chroma_collection()
    if collection.count() == 0:
        return []

    data = collection.get(include=["metadatas"])
    sources = {m.get("source", "unknown") for m in data.get("metadatas", [])}
    return sorted(sources)


def query_local_knowledge(topic: str):
    """
    Ask our local ChromaDB: "do we already know anything about this topic?"

    Returns a list of dicts: [{text, source, distance}, ...]
    Only chunks that pass the DISTANCE_THRESHOLD (i.e. are actually
    relevant) are returned. If nothing relevant is found, returns [].
    """
    collection = get_chroma_collection()

    # If nothing has ever been ingested, don't even bother querying.
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[topic],
        n_results=min(config.TOP_K_LOCAL_CHUNKS, collection.count()),
    )

    relevant_chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for text, meta, dist in zip(docs, metas, dists):
        # Lower distance = more relevant. Skip anything too far off-topic.
        if dist <= config.DISTANCE_THRESHOLD:
            relevant_chunks.append({
                "text": text,
                "source": meta.get("source", "local document"),
                "distance": round(dist, 3),
            })

    return relevant_chunks


def list_ingested_sources():
    """
    Returns the distinct document names currently stored in ChromaDB,
    with how many chunks each contributed. Used by the sidebar so you
    can actually see WHAT the agent has ingested, not just a chunk count.
    Returns: [{"source": "file.pdf", "chunks": 12}, ...]
    """
    collection = get_chroma_collection()
    if collection.count() == 0:
        return []

    data = collection.get(include=["metadatas"])
    counts = {}
    for meta in data["metadatas"]:
        name = meta.get("source", "unknown")
        counts[name] = counts.get(name, 0) + 1

    return [{"source": k, "chunks": v} for k, v in sorted(counts.items())]


def web_search(query: str):
    """
    Ask the live web (via Tavily) for information on a query.
    Returns a list of dicts: [{title, url, content}, ...]
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is missing - check your .env file")

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=config.TAVILY_MAX_RESULTS,
        search_depth="advanced",
    )

    results = []
    for item in response.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
        })
    return results
