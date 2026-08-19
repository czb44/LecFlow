from pathlib import Path

from sentence_transformers import SentenceTransformer

from .data import load_embed_model_ut


def predict_labels_embed_ut(embed_model, sent_transformer, sentences: list[str]) -> list[int]:
    """Predicts labels for sentences using trained embedding model"""
    sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    sentences_embedded = sent_transformer.encode(sentences)

    y_pred = embed_model.predict(sentences_embedded)

    return y_pred.tolist()  # np.array -> list


if __name__ == "__main__":
    unit_type_embed_model = load_embed_model_ut(Path("models"))
    sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")

    ex_sentences = [
        "For example, say we flip a coin ten times",
        (
            "Bayes Rule is the probability of an event based on prior knowledge, "
            "new evidence, and conditional probabilities."
        ),
        "What do you think?",
    ]
    ut_embed_predictions = predict_labels_embed_ut(
        unit_type_embed_model, sent_transformer, ex_sentences
    )

    for sentence, embed_pred in zip(ex_sentences, ut_embed_predictions):
        print(f"{embed_pred} | {sentence}")
