from lecflow.transcript import (
    filter_transcript,
    load_transcript,
    save_transcript,
    split_into_sentences,
)


def test_load_transcript(tmp_path):
    raw_transcript_text = "Today we will be learning Bayes Rule."

    file_path = tmp_path / "short_test_lecture.txt"
    file_path.write_text(raw_transcript_text)
    result = load_transcript(file_path)
    assert result == raw_transcript_text


def test_filter_transcript():
    test_str = "Lecture       One"
    result = filter_transcript(test_str)
    assert result == "Lecture One"


def test_split_into_sentences():
    test_sentences = [
        "In many problems we start with an initial idea, called a prior, and then gain new evidence.",
        "Suppose a disease is rare, affecting only one percent of the population.",
        "Conditional independence asks a different question: after we already know some additional information, do the two events still provide extra information about each other?",
        "This is extremely useful in applications ranging from medical diagnosis to spam filtering and machine learning.",
    ]
    test_block = (
        "In many problems we start with an initial idea, called a prior, and then gain new evidence. "
        "Suppose a disease is rare, affecting only one percent of the population. "
        "Conditional independence asks a different question: after we already know some additional information, do the two events still provide extra information about each other? "
        "This is extremely useful in applications ranging from medical diagnosis to spam filtering and machine learning."
    )
    result = split_into_sentences(test_block)
    assert result == test_sentences


def test_save_transcript(tmp_path):
    transcript = "Test transcript"
    out_path = tmp_path / "transcript_save_test.txt"
    save_transcript(transcript, out_path)
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == transcript
