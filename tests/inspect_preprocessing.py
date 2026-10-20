
import json
from pathlib import Path


INPUT_DIR = Path("data/processed/batch")


def inspect_documents():
    json_files = sorted(INPUT_DIR.glob("*.json"))

    if not json_files:
        raise FileNotFoundError("No processed JSON files found")

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            document = json.load(f)

        print("\n" + "=" * 60)
        print("DOCUMENT:", document["filename"])
        print("PAGES:", document["page_count"])
        print("CHUNKS:", document["chunk_count"])
        print("QUALITY:", document["quality"]["status"])

        for page in document["pages"]:
            print(f"\n--- PAGE {page['page_number']} ---")
            print("Method:", page["extraction_method"])
            print("Sections:", len(page["sections"]))

            for index, section in enumerate(page["sections"], 1):
                print(
                    f"\n{index}. Heading: {section['heading']}"
                )
                print("Type:", section["heading_type"])
                print(
                    "Content preview:",
                    " ".join(section["content"])[:120]
                )


if __name__ == "__main__":
    inspect_documents()
