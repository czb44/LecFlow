from lecflow.topics.cluster import sentence_embedding_cluster_train, group_by_cluster, get_topic_labels, get_candidate_phrases

def test_sentence_embedding_cluster_train(sent_transformer):
    content_sentences = ["Today we're continuing our introduction to probability by focusing on how we update our beliefs when we observe new information.", "By the end of today's lecture, you should understand Bayes' rule, the idea of conditional independence, how these concepts are applied in a worked example, and finally a few reminders about next week's assignment.", "Let's begin with Bayes' rule."]
    (cluster_model, cluster_labels) = sentence_embedding_cluster_train(content_sentences, sent_transformer, k=4)
    
    assert len(cluster_labels) == len(content_sentences)
    assert cluster_model.n_clusters == 3
    assert all(isinstance(label, int) for label in cluster_labels)



def test_group_by_cluster():
    content_sentences = ["Today we're continuing our introduction to probability by focusing on how we update our beliefs when we observe new information.", "By the end of today's lecture, you should understand Bayes' rule, the idea of conditional independence, how these concepts are applied in a worked example, and finally a few reminders about next week's assignment.", "Let's begin with Bayes' rule."]
    cluster_labels = [0, 1, 0]
    blocks = group_by_cluster(content_sentences, cluster_labels)

    assert blocks == {0: [content_sentences[0], content_sentences[2]], 1: [content_sentences[1]]}


def test_get_topic_labels(sent_transformer):
    content_sentences = ["Today we're continuing our introduction to probability by focusing on how we update our beliefs when we observe new information.", "By the end of today's lecture, you should understand Bayes' rule, the idea of conditional independence, how these concepts are applied in a worked example, and finally a few reminders about next week's assignment.", "Let's begin with Bayes' rule."]
    result = get_topic_labels(content_sentences, sent_transformer)

    assert isinstance(result, str)
    assert len(result) > 0

def test_get_candidate_phrases():
    sentences = ["Today we're continuing our introduction to probability by focusing on how we update our beliefs when we observe new information.", "By the end of today's lecture, you should understand Bayes' rule, the idea of conditional independence, how these concepts are applied in a worked example, and finally a few reminders about next week's assignment."]
    prediction = ['you', 'our introduction', 'the idea', 'our beliefs', 'we', 'a worked example', 'probability', 'these concepts', 'conditional independence', "Bayes' rule", "today's lecture", "next week's assignment", 'new information', 'the end']
    result = get_candidate_phrases(sentences)
    assert isinstance(result, list)
    assert len(result) > 0
    assert "probability" in result
    assert "conditional independence" in result
    assert "Bayes' rule" in result
