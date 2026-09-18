"""
RAG Retriever with FAISS + sentence-transformers
"""

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# PATHS
# --------------------------------------------------

INDEX_FILE = Path("data/knowledge_base/faiss/cardioagent.index")
METADATA_FILE = Path("data/knowledge_base/embeddings/metadata.json")

# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

# --------------------------------------------------
# LOAD (lazy)
# --------------------------------------------------

_model = None
_index = None
_metadata = None


def _load():
    """Load model, index, and metadata (lazy, cached)."""
    global _model, _index, _metadata
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(MODEL_NAME)
    if _index is None:
        print("Loading FAISS index...")
        _index = faiss.read_index(str(INDEX_FILE))
    if _metadata is None:
        print("Loading metadata...")
        _metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return _model, _index, _metadata


# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

def retrieve(question, top_k=5):
    """
    Retrieve relevant documents from FAISS index.

    Returns list of dicts with keys: chunk, source, score
    """
    model, index, metadata = _load()

    # Embed query
    query_embedding = model.encode([question], normalize_embeddings=True)
    query_embedding = np.asarray(query_embedding, dtype="float32")

    # Search
    scores, indices = index.search(query_embedding, top_k)

    # Build results
    results = []
    for score, index_number in zip(scores[0], indices[0]):
        if index_number == -1:
            continue
        record = metadata[int(index_number)]
        results.append({
            "chunk": record["text"],
            "source": record["source_file"],
            "score": float(score),
        })

    return results


if __name__ == "__main__":
    question = input("\nEnter your medical question: ")
    results = retrieve(question, top_k=5)
    print("\n" + "=" * 80)
    print("RETRIEVED RESULTS")
    print("=" * 80)
    for i, result in enumerate(results, start=1):
        print(f"\nRESULT {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"\nText:\n{result['chunk'][:500]}...")
        print("\n" + "-" * 80)
