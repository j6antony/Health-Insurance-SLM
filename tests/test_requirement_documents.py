
from src.verification.requirements import evaluate_requirement
from src.verification.requirement_rules import HOSPITALIZATION_REQUIREMENT


def test_requirement_documents():
    cases = [
        (
            [
                {
                    "document_type": "physician_statement",
                    "fields": {"diagnosis"}
                }
            ],
            True
        ),
        (
            [
                {
                    "document_type": "physician_statement",
                    "fields": set()
                }
            ],
            False
        ),
        (
            [
                {
                    "document_type": "ub04_form",
                    "fields": set()
                }
            ],
            True
        ),
        (
            [
                {
                    "document_type": "physician_statement",
                    "fields": set()
                },
                {
                    "document_type": "hospital_bill",
                    "fields": {"diagnosis"}
                }
            ],
            True
        )
    ]

    for documents, expected in cases:
        result = evaluate_requirement(
            HOSPITALIZATION_REQUIREMENT,
            documents
        )

        print(
            f"Documents: {documents} "
            f"-> Satisfied: {result['satisfied']}"
        )

        assert result["satisfied"] == expected

    print("\nDocument requirement test passed!")


if __name__ == "__main__":
    test_requirement_documents()
