
from pathlib import Path

from src.preprocessing.pipeline import preprocess_document


INPUT_DIR = Path("data/raw/reference_pdfs")
OUTPUT_DIR = Path("data/processed/batch")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_batch_preprocessing():
    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {INPUT_DIR}"
        )

    results = []

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        try:
            output_path = OUTPUT_DIR / f"{pdf_path.stem}.json"

            document = preprocess_document(
                file_path=pdf_path,
                output_path=output_path,
                max_words=300
            )

            results.append({
                "filename": pdf_path.name,
                "status": "success",
                "pages": document["page_count"],
                "chunks": document["chunk_count"],
                "quality": document["quality"]["status"]
            })

            print(
                f"Success | Pages: {document['page_count']} | "
                f"Chunks: {document['chunk_count']}"
            )

        except Exception as error:
            results.append({
                "filename": pdf_path.name,
                "status": "failed",
                "error": str(error)
            })

            print(f"Failed: {error}")

    print("\n--- Batch Processing Summary ---")

    for result in results:
        print(result)

    success_count = sum(
        result["status"] == "success"
        for result in results
    )

    print(f"\nSuccessful: {success_count}/{len(results)}")

    assert success_count == len(results), (
        "Some documents failed preprocessing"
    )


if __name__ == "__main__":
    test_batch_preprocessing()

