from lecflow import notes
from lecflow.notes import generate_notes, save_notes


def test_generate_notes(monkeypatch):
    #use monkeypatch to temporarily replace classify_unity_type w/ deterministic definition labeling
    monkeypatch.setattr(notes, "classify_unit_type", lambda sentence: "definition")
    blocks = {0: ["Conditional probability is the probability of event given another."]}
    topic_labels = {0: "Conditional Probability"}
    housekeeping_sentences = ["Next class is canceled."]
    result = generate_notes(blocks, topic_labels, housekeeping_sentences)
    assert "# Lecture Notes" in result
    assert "## Table of Contents" in result
    assert "- [Conditional Probability](#conditional-probability)" in result
    assert "- [Housekeeping](#housekeeping)" in result
    assert "## Conditional Probability" in result
    assert "**Definition:** Conditional probability is the probability of event given another." in result
    assert "## Housekeeping" in result
    assert "- Next class is canceled." in result


def test_save_notes(tmp_path):
    notes = "Test notes"
    out_path = tmp_path / "notes_save_test.txt"
    save_notes(notes, out_path)
    assert out_path.exists()
    assert out_path.read_text(encoding='utf-8') == notes
