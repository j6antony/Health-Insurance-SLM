
HOSPITALIZATION_REQUIREMENT = {
    "type": "OR",
    "requirements": [
        {
            "type": "document",
            "document_type": "ub04_form"
        },
        {
            "type": "document_with_fields",
            "document_type": "physician_statement",
            "fields": ["diagnosis"]
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
