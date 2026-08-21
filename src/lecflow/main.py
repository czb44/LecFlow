from pathlib import Path

from sentence_transformers import SentenceTransformer

from .classification.data import load_embed_model
from .classification.predict import predict_labels_embed
from .notes import generate_notes, save_notes
from .topics.cluster import (
    get_topic_labels,
    segment_by_similarity,
)
from .transcript import (
    filter_transcript,
    load_transcript,
    remove_short_artifacts,
    split_into_sentences,
)


def full_pipeline(raw_transcript_txt: str, housekeeping_model, sent_transformer) -> tuple[str, str]:
    """Runs full lecture processing pipeline. Returns filtered
    transcript and lecture notes."""
    # Load and clean transcript
    filtered = filter_transcript(raw_transcript_txt)
    raw_sentences = split_into_sentences(filtered)

    # Remove short artifacts / single word sentences
    sentences = remove_short_artifacts(raw_sentences)

    # Use pre-trained LR classifier (embed) for housekeeping
    embed_predictions = predict_labels_embed(housekeeping_model, sentences, sent_transformer)

    content_sentences = [
        sentence for sentence, prediction in zip(sentences, embed_predictions) if prediction == 0
    ]
    housekeeping_sentences = [
        sentence for sentence, prediction in zip(sentences, embed_predictions) if prediction == 1
    ]

    if len(content_sentences) < 3:
        blocks = {0: content_sentences}
    else:
        adj_blocks = segment_by_similarity(content_sentences, sent_transformer)

        blocks = {}
        for block_idx, block in enumerate(adj_blocks):
            blocks[block_idx] = block

    topic_labels = {}
    for cluster_num, group_of_sents in blocks.items():
        topic_labels[cluster_num] = get_topic_labels(group_of_sents, sent_transformer)

    notes = generate_notes(blocks, topic_labels, housekeeping_sentences)

    return filtered, notes


def main() -> None:
    file_path = Path("data/sample/6.006_Lecture_5.md")
    raw_transcript_txt = load_transcript(file_path)

    sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    housekeeping_model = load_embed_model(Path("models"))

    filtered, notes = full_pipeline(raw_transcript_txt, housekeeping_model, sent_transformer)

    output_path = Path("outputs/notes/6.006_Lecture_5_notes_notes_6.md")
    save_notes(notes, output_path)
    print(f"Notes saved to: {output_path}")


if __name__ == "__main__":
    main()
