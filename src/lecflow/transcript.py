from pathlib import Path

import spacy  # NLP to splice clean transcript


def load_transcript(file_path: Path) -> str:
    """Loads transcript and extracts its contents"""
    if file_path.exists():
        return file_path.read_text(encoding="utf-8")  # read all the same
    else:
        raise FileNotFoundError("File cannot be found. Verify path is correct")


def filter_transcript(transcript: str) -> str:
    """Formats extra whitespace in transcript with consistency"""
    trans_list = transcript.split()
    return " ".join(trans_list)


nlp = spacy.load("en_core_web_sm")  # use pre-trained English language model


def split_into_sentences(filtered_transcript: str) -> list[str]:
    """Split transcript into non-empty sentence strings using spacy NLP"""
    # Use NLP to find possible sentence boundaries
    proc_transcript = nlp(filtered_transcript)
    # Convert sentence to string, remove whitespace, and ignore blank / empty sentences
    return [sent.text.strip() for sent in proc_transcript.sents if sent.text.strip()]


def save_transcript(transcript: str, output_path: Path) -> None:
    """Ensures output folder exists and writes notes (Markdown string)
    to output Markdown File"""
    # create parent directories if neccessary, write notes to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(transcript, encoding="utf-8")


if __name__ == "__main__":
    file_path = Path("data/sample/lecture_1.txt")
    raw_transcript = load_transcript(file_path)
    filtered_transcript = filter_transcript(raw_transcript)
    sentences = split_into_sentences(filtered_transcript)
    print(sentences[:3])
