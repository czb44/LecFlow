from pathlib import Path
from sentence_transformers import SentenceTransformer
from .data import label_key, load_model, load_embed_model

def predict_labels(model, vectorizer, sentences: list[str]) -> list[int]:
    '''Predicts labels for new, unlabeled sentences using trained model and vectorizer'''
    #Vectorize raw sentences internally w/ saved vectorizer
    x_test_vec = vectorizer.transform(sentences)  
    
    y_pred = model.predict(x_test_vec)

    return y_pred.tolist() #np.array -> list

def predict_labels_embed(embed_model, sentences: list[str], sent_transformer: SentenceTransformer) -> list[int]:
    '''Predicts labels for new, unlabeled sentences using trained sentence-transformer embedding model'''
    sentences_embedded = sent_transformer.encode(sentences)
    
    y_pred = embed_model.predict(sentences_embedded)

    return y_pred.tolist() #np.array -> list


if __name__ == '__main__':
    model, vectorizer = load_model(Path('models'))
    embed_model = load_embed_model(Path('models'))
    sent_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    sentences = ['Next class is canceled because of the storm', 'Counting is a fundamental skill in probability']

    predictions = predict_labels(model, vectorizer, sentences)
    embed_predictions = predict_labels_embed(embed_model, sentences, sent_transformer)

    for sentence, tfidf_pred, embed_pred in zip(sentences, predictions, embed_predictions):
        print(f'TF-IDF: {label_key[tfidf_pred]} | Embedding: {label_key[embed_pred]} | {sentence}')
