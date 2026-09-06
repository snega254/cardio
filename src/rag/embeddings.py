from pathlib import Path
import json
import numpy as np

from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# FOLDERS
# --------------------------------------------------

CHUNK_DIR = Path(
    "data/knowledge_base/chunks"
)

EMBEDDING_DIR = Path(
    "data/knowledge_base/embeddings"
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    EMBEDDING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Loading embedding model: "
        f"{MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    json_files = list(
        CHUNK_DIR.glob("*_chunks.json")
    )

    if not json_files:

        print(
            "No chunk files found."
        )

        return

    all_records = []

    all_texts = []

    # --------------------------------------------------
    # LOAD ALL CHUNKS
    # --------------------------------------------------

    for json_file in json_files:

        print(
            f"Reading {json_file.name}"
        )

        records = json.loads(
            json_file.read_text(
                encoding="utf-8"
            )
        )

        all_records.extend(records)

        all_texts.extend(
            record["text"]
            for record in records
        )

    print(
        f"\nTotal chunks: "
        f"{len(all_texts)}"
    )

    # --------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------

    print(
        "\nCreating embeddings..."
    )

    embeddings = model.encode(
        all_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    np.save(
        EMBEDDING_DIR /
        "embeddings.npy",
        embeddings
    )

    metadata_file = (
        EMBEDDING_DIR /
        "metadata.json"
    )

    metadata_file.write_text(
        json.dumps(
            all_records,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        "\nEmbeddings saved."
    )

    print(
        f"Vector file: "
        f"{EMBEDDING_DIR / 'embeddings.npy'}"
    )

    print(
        f"Metadata file: "
        f"{metadata_file}"
    )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()