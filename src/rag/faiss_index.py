from pathlib import Path

import faiss
import numpy as np


# --------------------------------------------------
# PATHS
# --------------------------------------------------

EMBEDDING_DIR = Path(
    "data/knowledge_base/embeddings"
)

INDEX_DIR = Path(
    "data/knowledge_base/faiss"
)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    embedding_file = (
        EMBEDDING_DIR /
        "embeddings.npy"
    )

    # --------------------------------------------------
    # LOAD EMBEDDINGS
    # --------------------------------------------------

    embeddings = np.load(
        embedding_file
    ).astype("float32")

    print(
        f"Loaded embeddings: "
        f"{embeddings.shape}"
    )

    # --------------------------------------------------
    # CREATE FAISS INDEX
    # --------------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    # Add vectors
    index.add(embeddings)

    print(
        f"Vectors stored in FAISS: "
        f"{index.ntotal}"
    )

    # --------------------------------------------------
    # SAVE INDEX
    # --------------------------------------------------

    index_file = (
        INDEX_DIR /
        "cardioagent.index"
    )

    faiss.write_index(
        index,
        str(index_file)
    )

    print(
        f"FAISS index saved to:"
    )

    print(index_file)


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()