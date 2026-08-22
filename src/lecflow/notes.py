from pathlib import Path

from .unit_type.rules import classify_unit_type


def generate_notes(
    blocks: dict[int, list[str]],
    topic_labels: dict[int, str],
    housekeeping_sentences: list[str],
) -> str:
    """Converts cleaned transcript to a Markdown-formatted string"""
    heading = "# Lecture Notes\n\n"
    toc_header = "## Table of Contents\n\n"

    toc = ""
    body = ""

    # Display sentences of each block in natural lecture flow
    for i, block in blocks.items():
        label = topic_labels[i].title()
        link_case = label.strip().lower().replace("'", "").replace(" ", "-")
        body += f"## {label}\n\n"
        toc += f"- [{label}](#{link_case})\n"
        for sentence in block:
            unit_type = classify_unit_type(sentence)
            if unit_type == "example":
                body += f"- *Example:* {sentence}\n"
            elif unit_type == "question":
                body += f"- *Question:* {sentence}\n"
            elif unit_type == "definition":
                body += f"- **Definition:** {sentence}\n"
            elif unit_type == "explanation":  # no label for content
                body += f"- {sentence}\n"
        body += "\n\n"

    if housekeeping_sentences:  # Only add if relevant content remains
        toc += "- [Housekeeping](#housekeeping)\n\n\n"
        housekeeping_head = "## Housekeeping\n\n"
        housekeeping_body = ""
        for sent in housekeeping_sentences:
            housekeeping_body += f"- {sent}\n"
    else:
        housekeeping_head = ""
        housekeeping_body = ""

    return heading + toc_header + toc + body + housekeeping_head + housekeeping_body


def housekeeping_only_notes(housekeeping_sentences: list[str]) -> str:
    """Creates lecture notes and formats housekeeping sentences
    in the case where the lecture contains no content sentences"""
    heading = "# Lecture Notes\n\n"
    if housekeeping_sentences:  # Only add if relevant content remains
        housekeeping_head = "## Housekeeping\n\n"
        housekeeping_body = ""
        for sent in housekeeping_sentences:
            housekeeping_body += f"- {sent}\n"
    else:
        housekeeping_head = ""
        housekeeping_body = ""

    return heading + housekeeping_head + housekeeping_body


def save_notes(notes: str, output_path: Path) -> None:
    """Ensures output folder exists and writes notes (Markdown string)
    to output Markdown File"""
    # create parent directories if neccessary, write notes to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(notes, encoding="utf-8")
