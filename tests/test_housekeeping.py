from lecflow.classification.predict import predict_labels_embed


def test_predict_labels_embed(housekeeping_model, sent_transformer):
    predictions = [0, 0, 1, 1] #content = 0, housekeeping = 1
    sentences = ["Notice that the probability does not become ninety-five percent simply because the word is common in spam.", 
    "The base rate of spam still matters, and the evidence must always be interpreted in the context of the prior information.", 
    "Before we finish, a few housekeeping announcements", 
    "The homework on conditional probability is due next Tuesday at 11:59 PM."]
    result = predict_labels_embed(housekeeping_model, sentences, sent_transformer)
    assert result == predictions
