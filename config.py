# config.py
# ------------------------------------------------------------------
# All the "tunable knobs" of Noesis AI live here in one place.
# If you want to change how strict/loose the pipeline behaves,
# change a number here instead of hunting through the code.
# ------------------------------------------------------------------

# --- LLM settings (Groq) ---
GROQ_MODEL = "llama-3.3-70b-versatile"   # fast + good enough for report writing
LLM_TEMPERATURE = 0.3                     # low = more focused, less random

# --- ChromaDB (local knowledge base) settings ---
CHROMA_DB_PATH = "chroma_db"              # folder where vectors are stored on disk
CHROMA_COLLECTION_NAME = "noesis_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"      # small, fast, runs locally (Sentence-Transformers)

TOP_K_LOCAL_CHUNKS = 4                    # how many chunks to pull from ChromaDB per query
# ChromaDB returns "distance", NOT similarity. Smaller distance = more relevant.
# 0.0 = identical meaning, ~1.0+ = unrelated. We only trust chunks under this value.
DISTANCE_THRESHOLD = 0.9

# --- Web search (Tavily) settings ---
TAVILY_MAX_RESULTS = 4

# --- Self-critique loop settings ---
CRITIQUE_PASS_SCORE = 7.0     # draft is "good enough" once critic gives this score or higher
MAX_REFINE_ITERATIONS = 2     # never loop forever - hard stop after this many rewrites

# --- Claim verification settings ---
MAX_CLAIMS_TO_VERIFY = 4      # only check the N most important claims (keeps it fast/cheap)

# --- Document chunking (used during ingestion) ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
