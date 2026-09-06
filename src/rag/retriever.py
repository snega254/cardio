from pathlib import Path
import json

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PATHS
# --------------------------------------------------

INDEX_FILE = Path(
    "data/knowledge_base/faiss/cardioagent.index"
)

METADATA_FILE = Path(
    "data/knowledge_base/embeddings/metadata.json"
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"


# --------------------------------------------------
# LOAD EVERYTHING
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Loading FAISS index...")

index = faiss.read_index(
    str(INDEX_FILE)
)

print("Loading metadata...")

metadata = json.loads(
    METADATA_FILE.read_text(
        encoding="utf-8"
    )
)


# --------------------------------------------------
# SEARCH FUNCTION
# --------------------------------------------------

def retrieve(
    question,
    top_k=5
):

    # Convert question to vector
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Search
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):

        if index_number == -1:
            continue

        record = metadata[
            int(index_number)
        ]

        results.append({

            "score":
                float(score),

            "chunk_id":
                record["chunk_id"],

            "source_file":
                record["source_file"],

            "text":
                record["text"]
        })

    return results


# --------------------------------------------------
# TEST RETRIEVER
# --------------------------------------------------

if __name__ == "__main__":

    question = input(
        "\nEnter your medical question: "
    )

    results = retrieve(
        question,
        top_k=5
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RETRIEVED RESULTS"
    )

    print(
        "=" * 80
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT {i}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Source: "
            f"{result['source_file']}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            "\nText:"
        )

        print(
            result["text"][:1500]
        )

        print(
            "\n" + "-" * 80
        )