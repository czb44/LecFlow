from lecflow.topics.cluster import (
    find_best_k,
    get_candidate_phrases,
    get_topic_labels,
    group_by_cluster,
    sentence_embedding_cluster_train,
)


def test_sentence_embedding_cluster_train(sent_transformer):
    content_sentences = [
        (
            "Today we're continuing our introduction to probability by "
            "focusing on how we update our beliefs when we observe new information."
        ),
        (
            "By the end of today's lecture, you should understand Bayes' rule, "
            "the idea of conditional independence, "
            "how these concepts are applied in a worked example, "
            "and finally a few reminders about next week's assignment."
        ),
        "Let's begin with Bayes' rule.",
    ]
    (cluster_model, cluster_labels) = sentence_embedding_cluster_train(
        content_sentences, sent_transformer, k=3
    )

    assert len(cluster_labels) == len(content_sentences)
    assert cluster_model.n_clusters == 3
    assert all(isinstance(label, int) for label in cluster_labels)


def test_group_by_cluster():
    content_sentences = [
        (
            "Today we're continuing our introduction to probability by "
            "focusing on how we update our beliefs when we observe new information."
        ),
        (
            "By the end of today's lecture, you should understand Bayes' rule, "
            "the idea of conditional independence, "
            "how these concepts are applied in a worked example, "
            "and finally a few reminders about next week's assignment."
        ),
        "Let's begin with Bayes' rule.",
    ]
    cluster_labels = [0, 1, 0]
    blocks = group_by_cluster(content_sentences, cluster_labels)

    assert blocks == {
        0: [content_sentences[0], content_sentences[2]],
        1: [content_sentences[1]],
    }


def test_get_topic_labels(sent_transformer):
    content_sentences = [
        (
            "Today we're continuing our introduction to probability by "
            "focusing on how we update our beliefs when we observe new information."
        ),
        (
            "By the end of today's lecture, you should understand Bayes' rule, "
            "the idea of conditional independence, "
            "how these concepts are applied in a worked example, "
            "and finally a few reminders about next week's assignment."
        ),
        "Let's begin with Bayes' rule.",
    ]
    result = get_topic_labels(content_sentences, sent_transformer)

    assert isinstance(result, str)
    assert len(result) > 0


def test_get_candidate_phrases():
    sentences = [
        (
            "Today we're continuing our introduction to probability by "
            "focusing on how we update our beliefs when we observe new information."
        ),
        (
            "By the end of today's lecture, you should understand Bayes' rule, "
            "the idea of conditional independence, "
            "how these concepts are applied in a worked example, "
            "and finally a few reminders about next week's assignment."
        ),
        "Let's begin with Bayes' rule.",
    ]
    result = get_candidate_phrases(sentences)
    assert isinstance(result, list)
    assert len(result) > 0
    assert "probability" in result
    assert "conditional independence" in result
    assert "Bayes' rule" in result


def test_find_best_k_small(sent_transformer):
    sentences = [
        "Biology is the study of life.",
        "Some cells have organelles.",
    ]

    embeddings = sent_transformer.encode(sentences)

    result = find_best_k(embeddings)

    assert result == 1


def test_find_best_k_range(sent_transformer):
    sentences = [
        "Bayes rule updates prior probabilities.",
        "Posterior probability incorporates new evidence.",
        "Conditional independence simplifies probability models.",
        "Hash tables support fast lookup.",
        "Hash functions map keys to array locations.",
        "Sorting algorithms arrange elements into order.",
        "DNA is genetic information.",
        "Genes encode biological information.",
        "Proteins are produced from genetic instructions.",
        "Mutations can change DNA sequences.",
        "Probability models uncertainty.",
        "Random variables describe uncertain outcomes.",
    ]

    embeddings = sent_transformer.encode(sentences)

    result = find_best_k(embeddings)

    assert isinstance(result, int)
    assert 1 <= result < len(sentences)


def test_sentence_embedding_cluster_train_best_k(sent_transformer):
    sentences = [
        "Bayes rule updates prior probabilities.",
        "Posterior probability incorporates new evidence.",
        "Conditional independence simplifies probability models.",
        "Hash tables support fast lookup.",
        "Hash functions map keys to array locations.",
        "Sorting algorithms arrange elements into order.",
        "DNA is genetic information.",
        "Genes encode biological information.",
        "Proteins are produced from genetic instructions.",
        "Mutations can change DNA sequences.",
        "Probability models uncertainty.",
        "Random variables describe uncertain outcomes.",
    ]
    (model, cluster_labels) = sentence_embedding_cluster_train(sentences, sent_transformer)
    assert len(cluster_labels) == len(sentences)
    assert isinstance(model.n_clusters, int)
    assert 1 <= model.n_clusters < len(sentences)
