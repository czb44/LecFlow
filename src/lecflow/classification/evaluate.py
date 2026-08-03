import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from ..data import load_data, split_sentences_labels
from.data import label_key, load_model, load_embed_model
data_labels = [label_key[0], label_key[1]]

def show_misclassified(x_test: list[str], y_test: list[int], y_pred: list[int], num_examples: int = 50) -> None:
    '''Prints up to num_examples sentences the model evaluated incorrectly'''
    ct = 0
    for i in range(len(x_test)):
        if ct >= num_examples:
            break
        sentence = x_test[i]
        true = y_test[i]
        predicted = y_pred[i]

        if true != predicted:
            print(f"Sentence: {sentence}")
            print(f"True: {label_key[true]}")
            print(f"Predicted: {label_key[predicted]}")
            print()
            ct += 1

def evaluate_model(model, vectorizer, x_test, y_test, output_path) -> None:
    '''Analyze model performance with accuracy, classification report, 
    and confusion matrix. Plots and saves confusion matrix to output path'''
    #Vectorize raw sentences internally w/ saved vectorizer
    x_test_vec = vectorizer.transform(x_test)  
    
    y_pred = model.predict(x_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    
    
    cr = classification_report(y_test, y_pred, target_names=data_labels)
    print(cr) #Precision, recall, F1

    show_misclassified(x_test, y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    graph = ConfusionMatrixDisplay(cm,display_labels=data_labels)
    graph.plot()
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.show()
    plt.close() 

def evaluate_embed_model(embed_model, x_test, y_test, output_path) -> None:
    '''Analyze sentence-trasnsformer embedding model performance with accuracy, classification report, 
    and confusion matrix. Plots and saves confusion matrix to output path'''
    embedding_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    sentences_embedded = embedding_transformer.encode(x_test)

    y_pred = embed_model.predict(sentences_embedded)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    
    
    cr = classification_report(y_test, y_pred, target_names=data_labels)
    print(cr) #Precision, recall, F1

    show_misclassified(x_test, y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    graph = ConfusionMatrixDisplay(cm,display_labels=data_labels)
    graph.plot()
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.show()
    plt.close() 


if __name__ == '__main__':
    model, vectorizer = load_model(Path('models'))
    embed_model = load_embed_model(Path('models'))

    hard_data = load_data(Path('data/labeled/hard_housekeeping_dataset.csv'))
    x_hard, y_hard = split_sentences_labels(hard_data)

    output_path = Path('outputs/confusion_matrix_hard_normal.png')
    evaluate_model(model, vectorizer, x_hard, y_hard, output_path)

    output_path = Path('outputs/confusion_matrix_hard_embed.png')
    evaluate_embed_model(embed_model, x_hard, y_hard, output_path)
