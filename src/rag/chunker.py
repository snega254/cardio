from pathlib import Path
import json
import re


# --------------------------------------------------
# FOLDERS
# --------------------------------------------------

CLEAN_DIR = Path(
    "data/knowledge_base/cleaned"
)

CHUNK_DIR = Path(
    "data/knowledge_base/chunks"
)


# --------------------------------------------------
# SPLIT TEXT INTO PARAGRAPHS
# --------------------------------------------------

def split_into_paragraphs(text):

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    paragraphs = [
        p.strip()
        for p in paragraphs
        if p.strip()
    ]

    return paragraphs


# --------------------------------------------------
# CREATE CHUNKS
# --------------------------------------------------

def create_chunks(
    text,
    chunk_size=400,
    overlap=80
):

    paragraphs = split_into_paragraphs(
        text
    )

    chunks = []

    current_words = []

    for paragraph in paragraphs:

        words = paragraph.split()

        # If adding this paragraph keeps
        # the chunk reasonably small
        if (
            len(current_words) + len(words)
            <= chunk_size
        ):

            current_words.extend(words)

        else:

            if current_words:

                chunks.append(
                    " ".join(current_words)
                )

            # Keep overlap
            overlap_words = (
                current_words[-overlap:]
                if len(current_words) > overlap
                else current_words
            )

            current_words = (
                overlap_words + words
            )

    # Add final chunk
    if current_words:

        chunks.append(
            " ".join(current_words)
        )

    return chunks


# --------------------------------------------------
# PROCESS FILES
# --------------------------------------------------

def main():

    CHUNK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    text_files = list(
        CLEAN_DIR.glob("*.txt")
    )

    if not text_files:

        print(
            "No cleaned text files found."
        )

        return

    for text_file in text_files:

        print(
            f"\nChunking: {text_file.name}"
        )

        text = text_file.read_text(
            encoding="utf-8"
        )

        chunks = create_chunks(
            text,
            chunk_size=400,
            overlap=80
        )

        records = []

        for index, chunk in enumerate(
            chunks
        ):

            record = {

                "chunk_id":
                    f"{text_file.stem}_{index:05d}",

                "source_file":
                    text_file.name,

                "chunk_index":
                    index,

                "text":
                    chunk
            }

            records.append(record)

        output_file = (
            CHUNK_DIR /
            f"{text_file.stem}_chunks.json"
        )

        output_file.write_text(
            json.dumps(
                records,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        print(
            f"Created {len(records)} chunks"
        )

        print(
            f"Saved: {output_file}"
        )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()