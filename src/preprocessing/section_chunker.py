def chunk_sections(sections, max_words=300):
    if type(max_words) is not int or max_words <= 0:
        raise ValueError("max_words must be a positive integer")
    chunks = []

    for section_index, section in enumerate(sections):
        heading = section["heading"]
        lines = section["source_lines"]

        current_lines = []
        heading_words = len((heading or "").split())
        preserve_alternatives = any(line["text"].strip().upper() == "OR" for line in lines)
        current_word_count = heading_words
        chunk_index = 0

        def save_chunk():
            nonlocal current_lines, current_word_count, chunk_index

            if not current_lines and not (heading and chunk_index == 0):
                return

            chunk_index += 1

            content = "\n".join(line["text"] for line in current_lines)

            chunks.append({
                "chunk_id": f"section_{section_index}_chunk_{chunk_index}",
                "heading": heading,
                "heading_type": section["heading_type"],
                "heading_line_id": section["heading_line_id"],
                "text": (
                    f"{heading}\n{content}"
                    if heading else content
                ),
                "source_lines": [
                    line["line_id"] for line in current_lines
                ],
                "word_count": current_word_count,
                "oversized": current_word_count > max_words

            })
            current_lines = []
            current_word_count = heading_words

        groups = []
        for line in lines:
            same_paragraph = (groups and line.get("paragraph_id") is not None
                              and groups[-1][-1].get("paragraph_id") == line["paragraph_id"])
            continuation = (groups and line["text"].strip()
                            and line["text"].strip()[0].islower())
            if same_paragraph or continuation:
                groups[-1].append(line)
            else:
                groups.append([line])

        for group in groups:
            group_words = sum(len(line["text"].split()) for line in group)
            if (current_lines and not preserve_alternatives
                    and current_word_count + group_words > max_words):
                save_chunk()
            current_lines.extend(group)
            current_word_count += group_words
        save_chunk()

    return chunks
