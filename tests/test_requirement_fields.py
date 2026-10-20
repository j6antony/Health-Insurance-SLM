
from src.verification.requirements import evaluate_requirement
from src.verification.requirement_rules import HOSPITALIZATION_REQUIREMENT

RULE = {
    "type": "OR",
    "requirements": [
        {
            "type": "document",
            "document_type": "ub04_form"
        },
        {
            "type": "AND",
            "requirements": [
                {
                    "type": "document",
                    "document_type": "physician_statement"
                },
                {
                    "type": "field",
                    "field_name": "diagnosis"
                }
            ]
        },
        {
            "type": "AND",
            "requirements": [
                {
                    "type": "document",
                    "document_type": "room_board_documentation"
                },
                {
                    "type": "document",
                    "document_type": "medical_codes_documentation"
                }
            ]
        }
    ]
}


def test_requirement_fields():
    cases = [
        (
            {"ub04_form"},
            set(),
            True
        ),
        (
            {"physician_statement"},
            {"diagnosis"},
            True
        ),
        (
            {"physician_statement"},
            set(),
            False
        ),
        (
            {
                "room_board_documentation",
                "medical_codes_documentation"
            },
            set(),
            True
        ),
        (
            {"room_board_documentation"},
            {"diagnosis"},
            False
        )
    ]

    for documents, fields, expected in cases:
        result = evaluate_requirement(
            HOSPITALIZATION_REQUIREMENT,
            documents,
            fields
        )

        print(
            f"Documents: {documents} | "
            f"Fields: {fields} | "
            f"Satisfied: {result['satisfied']}"
        )

        assert result["satisfied"] == expected

    print("\nRequirement field verification passed!")


if __name__ == "__main__":
    test_requirement_fields()
