
import re


def recover_evidence(value, source_text, window=80):
    """
    Recover a verbatim evidence excerpt around an exact
    field value in the source text.

    Returns None when the value cannot be located.
    """

    if not value or not value.strip():
        return None

    matches = list(re.finditer(
        re.escape(value),
        source_text
    ))

    # Do not guess between multiple occurrences.
    if len(matches) != 1:
        return None

    match = matches[0]

    start = max(0, match.start() - window)
    end = min(len(source_text), match.end() + window)

    return source_text[start:end].strip()
