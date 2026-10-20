
from src.verification.requirements import evaluate_requirement
from src.verification.requirement_rules import HOSPITALIZATION_REQUIREMENT


def test_document_scoped_requirements():
    cases = [
        (
            "Completed physician statement",
            [
                {
                    "document_type": "physician_statement",
                    "fields": {"diagnosis"}
                }
            ],
            True
        ),
        (
            "Incomplete physician statement",
            [
                {
                    "document_type": "physician_statement",
                    "fields": set()
                }
            ],
            False
        ),
        (
            "Diagnosis only in hospital bill",
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
            False
        ),
        (
            "UB-04 provided",
            [
                {
                    "document_type": "ub04_form",
                    "fields": set()
                }
            ],
            True
        ),
        (
            "Both supporting documents",
            [
                {
                    "document_type": "room_board_documentation",
                    "fields": set()
                },
                {
                    "document_type": "medical_codes_documentation",
                    "fields": set()
                }
            ],
            True
        ),
        (
            "Only one supporting document",
            [
                {
                    "document_type": "room_board_documentation",
                    "fields": set()
                }
            ],
            False
        )
    ]

    for name, documents, expected in cases:
        result = evaluate_requirement(
            HOSPITALIZATION_REQUIREMENT,
            documents
        )

        print(
            f"{name}: "
            f"{result['satisfied']} "
            f"(Expected: {expected})"
        )

        assert result["satisfied"] == expected

    print("\nDocument-scoped requirements test passed!")


if __name__ == "__main__":
    test_document_scoped_requirements()
