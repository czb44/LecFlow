from pathlib import Path
from .data import label_key, load_model

def predict_labels(model, vectorizer, sentences: list[str]) -> list[int]:
    '''Predicts labels for new, unlabeled sentences using trained model and vectorizer'''
    #Vectorize raw sentences internally w/ saved vectorizer
    x_test_vec = vectorizer.transform(sentences)  
    
    y_pred = model.predict(x_test_vec)

    return y_pred.tolist() #np.array -> list



if __name__ == '__main__':
    model, vectorizer = load_model(Path('models'))

    sentences = ['Next class is canceled because of the storm', 'Counting is a fundamental skill in probability']

    predictions = predict_labels(model, vectorizer, sentences)

    for sentence, label in zip(sentences, predictions):
        print(f'{label_key[label]}({label}): {sentence}')
