from pathlib import Path
import re
from pypdf import PdfReader


# --------------------------------------------------
# FOLDER LOCATIONS
# --------------------------------------------------

RAW_DIR = Path("data/knowledge_base/raw_sources")
CLEAN_DIR = Path("data/knowledge_base/cleaned")


# --------------------------------------------------
# EXTRACT TEXT FROM ONE PDF
# --------------------------------------------------

def extract_pdf_text(pdf_path):
    """
    Extract text from every page of a PDF.
    """

    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text()

            if text:
                pages.append(text)

        except Exception as error:
            print(
                f"Could not read page {page_number} "
                f"of {pdf_path.name}: {error}"
            )

    return "\n\n".join(pages)


# --------------------------------------------------
# CLEAN EXTRACTED TEXT
# --------------------------------------------------

def clean_text(text):
    """
    Perform basic cleaning while preserving
    the medical content.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Join words broken by hyphen at line break
    text = re.sub(r"-\n", "", text)

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove excessive spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Replace single newlines with spaces
    # while preserving paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Reduce excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# PROCESS ALL PDF FILES
# --------------------------------------------------

def main():

    # Make sure cleaned directory exists
    CLEAN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = list(
        RAW_DIR.glob("*.pdf")
    )

    if not pdf_files:

        print(
            "No PDF files found in:"
        )

        print(RAW_DIR)

        return

    print(
        f"\nFound {len(pdf_files)} PDF files.\n"
    )

    for pdf_path in pdf_files:

        print(
            f"Processing: {pdf_path.name}"
        )

        # Extract
        raw_text = extract_pdf_text(
            pdf_path
        )

        # Clean
        cleaned_text = clean_text(
            raw_text
        )

        # Output filename
        output_path = (
            CLEAN_DIR /
            f"{pdf_path.stem}.txt"
        )

        # Save
        output_path.write_text(
            cleaned_text,
            encoding="utf-8"
        )

        print(
            f"Saved: {output_path}"
        )

        print(
            f"Characters: "
            f"{len(cleaned_text):,}"
        )

        print("-" * 60)


# --------------------------------------------------
# RUN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":
    main()