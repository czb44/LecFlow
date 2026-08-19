from pathlib import Path

import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from ..data import load_data, split_sentences_labels
from .data import load_embed_model_ut, unit_type_labels

unit_type_labels_key = [
    unit_type_labels[0],
    unit_type_labels[1],
    unit_type_labels[2],
    unit_type_labels[3],
]


def show_misclassified(
    x_test: list[str], y_test: list[int], y_pred: list[int], num_examples: int = 50
) -> None:
    """Prints up to num_examples sentences the model evaluated incorrectly"""
    ct = 0
    for i in range(len(x_test)):
        if ct >= num_examples:
            break
        sentence = x_test[i]
        true = y_test[i]
        predicted = y_pred[i]

        if true != predicted:
            print(f"Sentence: {sentence}")
            print(f"True: {true}")
            print(f"Predicted: {predicted}")
            print()
            ct += 1


def evaluate_embed_model(embed_model, sent_transformer, x_test, y_test, output_path) -> None:
    """Analyze sentence-transformer embedding model performance with accuracy,
    classification report, and confusion matrix.
    Plots and saves confusion matrix to output path"""
    sentences_embedded = sent_transformer.encode(x_test)

    y_pred = embed_model.predict(sentences_embedded)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    cr = classification_report(y_test, y_pred, target_names=unit_type_labels_key)
    print(cr)  # Precision, recall, F1

    show_misclassified(x_test, y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    graph = ConfusionMatrixDisplay(cm, display_labels=unit_type_labels_key)
    graph.plot()
    plt.title("Confusion Matrix")
    plt.savefig(output_path)
    plt.show()
    plt.close()


if __name__ == "__main__":
    embed_model = load_embed_model_ut(Path("models"))
    sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")

    data = load_data(Path("data/splits/gold_unit_type_test_set.csv"))
    x_test, y_test = split_sentences_labels(data)

    output_path = Path("outputs/confusion_matrix_gold_unit_type.png")
    evaluate_embed_model(embed_model, sent_transformer, x_test, y_test, output_path)
