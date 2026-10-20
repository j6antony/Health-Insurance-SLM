
from src.verification.requirements import evaluate_requirement


RULE = {
    "type": "OR",
    "requirements": [
        {
            "type": "document",
            "document_type": "hospital_bill"
        },
        {
            "type": "document",
            "document_type": "physician_statement"
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


def test_requirements():
    cases = [
        (
            {"hospital_bill"},
            True
        ),
        (
            {"physician_statement"},
            True
        ),
        (
            {
                "room_board_documentation",
                "medical_codes_documentation"
            },
            True
        ),
        (
            {"room_board_documentation"},
            False
        ),
        (
            set(),
            False
        )
    ]

    for available, expected in cases:
        result = evaluate_requirement(
            RULE,
            available
        )

        print(
            f"Available: {available} "
            f"-> Satisfied: {result['satisfied']}"
        )

        assert result["satisfied"] == expected

    print("\nRequirement verification test passed!")


if __name__ == "__main__":
    test_requirements()
