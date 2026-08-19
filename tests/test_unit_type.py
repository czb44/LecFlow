from lecflow.unit_type.rules import classify_unit_type


def test_classify_unit_type():
    prediction = ["definition", "example", "question", "explanation"]
    result = []
    test_sentences = [
        (
            "In many problems we start with an initial idea, called a prior, "
            "and then gain new evidence."
        ),
        "Suppose a disease is rare, affecting only one percent of the population.",
        (
            "Conditional independence asks a different question: "
            "after we already know some additional information, "
            "do the two events still provide extra information about each other?"
        ),
        (
            "This is extremely useful in applications ranging from "
            "medical diagnosis to spam filtering and machine learning."
        ),
    ]
    for sent in test_sentences:
        result.append(classify_unit_type(sent))
    assert result == prediction
